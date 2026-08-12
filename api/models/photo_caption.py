import os
from django.db import models
from django.db.models import Q

import api.models
from api import util
import requests
import time

# --- Configuration (from Environment Variables) ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")
CAPTION_GENERATOR_HOST = os.getenv("CAPTION_GENERATOR_HOST", "caption-generator")
CAPTION_GENERATOR_PORT = int(os.getenv("CAPTION_GENERATOR_PORT", 8020))
CAPTION_GENERATOR_API_ENDPOINT = os.getenv("CAPTION_GENERATOR_API_ENDPOINT", "rushgenerate")
CAPTION_GENERATOR_HEALTH_ENDPOINT = os.getenv("CAPTION_GENERATOR_HEALTH_ENDPOINT", "health")
CAPTION_GENERATOR_TIMEOUT_SEC = int(os.getenv("CAPTION_GENERATOR_TIMEOUT_SEC", 300))
CAPTION_GENERATOR_RETRIES = int(os.getenv("CAPTION_GENERATOR_RETRIES", 5))
CAPTION_GENERATOR_RETRY_BACKOFF = float(os.getenv("CAPTION_GENERATOR_RETRY_BACKOFF", 2.0))
CAPTION_GENERATOR_STARTUP_TIMEOUT_SEC = int(os.getenv("CAPTION_GENERATOR_STARTUP_TIMEOUT_SEC", 120))

def ensure_caption_generator_ready() -> bool:
    """
    Ensure caption-generator container is running and health endpoint is reachable.
    """
    try:
        import docker
    except ImportError:
        docker = None

    health_url = f"http://{CAPTION_GENERATOR_HOST}:{CAPTION_GENERATOR_PORT}/{CAPTION_GENERATOR_HEALTH_ENDPOINT}"

    # Fast path: already up
    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            util.logger.info("caption-generator already healthy at %s", health_url)
            return True
    except Exception:
        util.logger.info("caption-generator is not reachable yet, attempting startup")

    if docker is None:
        util.logger.error(
            "Docker SDK is not installed; cannot start caption-generator container"
        )
        return False

    try:
        client = docker.from_env()
        container = client.containers.get(CAPTION_GENERATOR_HOST)

        container.reload()
        status = container.status
        util.logger.info(
            "caption-generator container '%s' current status: %s",
            CAPTION_GENERATOR_HOST,
            status,
        )

        if status != "running":
            util.logger.info(
                "Starting caption-generator container '%s'",
                CAPTION_GENERATOR_HOST,
            )
            container.start()
    except Exception as e:
        util.logger.error(
            "Failed to start caption-generator container '%s': %s",
            CAPTION_GENERATOR_HOST,
            e,
            exc_info=True,
        )
        return False

    deadline = time.time() + CAPTION_GENERATOR_STARTUP_TIMEOUT_SEC
    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                util.logger.info("caption-generator is healthy and ready")
                return True
        except Exception:
            pass

        time.sleep(CAPTION_GENERATOR_RETRY_BACKOFF)

    util.logger.error(
        "caption-generator did not become ready within %s seconds",
        CAPTION_GENERATOR_STARTUP_TIMEOUT_SEC,
    )
    return False

def generate_image_caption(image_path: str, file_ext: str):
    """
    Generates a caption by sending the image to the caption-generator via HTTP request.
    """
    CAPTION_GENERATOR_API_URL = f"http://{CAPTION_GENERATOR_HOST}:{CAPTION_GENERATOR_PORT}/{CAPTION_GENERATOR_API_ENDPOINT}"

    try:
        if not ensure_caption_generator_ready():
            util.logger.error(
                "caption-generator is unavailable; cannot generate caption for %s",
                image_path,
            )
            return None, None
        
        payload = { 
            "file_path": image_path, 
            "file_ext": file_ext 
        }

        attempts = max(CAPTION_GENERATOR_RETRIES, 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                util.logger.info(
                    "Sending caption request to %s (attempt %d/%d, timeout=%ss)",
                    CAPTION_GENERATOR_API_URL,
                    attempt,
                    attempts,
                    CAPTION_GENERATOR_TIMEOUT_SEC,
                )
                response = requests.post(
                    CAPTION_GENERATOR_API_URL,
                    json=payload,
                    timeout=CAPTION_GENERATOR_TIMEOUT_SEC,
                )

                if response.status_code == 200:
                    result = response.json()
                    caption_raw = (result.get("caption") or "").strip()

                    # Parse embedded objects/texts from caption string
                    caption = caption_raw
                    objects = ""
                    texts = ""
                    if "\nobjects:" in caption_raw:
                        parts = caption_raw.split("\nobjects:", 1)
                        caption = parts[0].strip()
                        remainder = parts[1]
                        if "\ntexts:" in remainder:
                            obj_part, txt_part = remainder.split("\ntexts:", 1)
                            objects = obj_part.strip()
                            texts = txt_part.strip()
                        else:
                            objects = remainder.strip()
                    elif "\ntexts:" in caption_raw:
                        parts = caption_raw.split("\ntexts:", 1)
                        caption = parts[0].strip()
                        texts = parts[1].strip()

                    # Fall back to separate API fields
                    if not objects:
                        objects = (result.get("objects") or "").strip()
                    if not texts:
                        texts = (result.get("texts") or "").strip()

                    tag_parts = [p for p in (objects, texts) if p]
                    tag = ", ".join(tag_parts) if tag_parts else None
                    if caption or tag:
                        util.logger.info(f"Generated caption for {image_path}: '{caption}', tag: {tag}")
                        return caption, tag
                    util.logger.error("Caption API returned empty response for %s", image_path)
                elif response.status_code == 504:
                    util.logger.warning(f"Server returned {response.status_code} (Processing) for {image_path}. Triggering retry...")
                    raise requests.exceptions.Timeout(f"Server returned {response.status_code} Gateway Timeout")
                else:
                    try:
                        err_msg = response.json()
                    except Exception:
                        err_msg = response.text
                    util.logger.error(f"API Error {response.status_code}: {err_msg}")
                    raise requests.exceptions.Timeout(f"Server returned {response.status_code}. Triggering retry...")
            except requests.exceptions.Timeout:
                if attempt >= attempts:
                    util.logger.error("Caption request timed out after %d attempt(s) for %s", attempts, image_path)
                sleep_s = CAPTION_GENERATOR_RETRY_BACKOFF * (2 ** (attempt - 1))
                util.logger.warning(
                    "Caption request timeout for %s; retrying in %.1fs (attempt %d/%d)",
                    image_path,
                    sleep_s,
                    attempt,
                    attempts,
                )
                time.sleep(sleep_s)
            except Exception as e:
                util.logger.error(f"Failed to generate caption for {image_path}: {e}")
                break
    except Exception as e:
        util.logger.error(f"Failed to generate caption for {image_path}: {e}", exc_info=True)
        pass
    finally:
        import gc
        import torch
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return None, None

class PhotoCaption(models.Model):
    """Model for handling image captions and related functionality"""

    photo = models.OneToOneField(
        "Photo",
        on_delete=models.CASCADE,
        related_name="caption_instance",
        primary_key=True,
    )
    captions_json = models.JSONField(blank=True, null=True, db_index=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "api_photo_caption"

    def __str__(self):
        return f"Captions for {self.photo.image_hash}"

    def generate_captions_im2txt(self, commit=True):
        """Generate im2txt captions for the photo"""
        if not self.photo.thumbnail or not self.photo.thumbnail.thumbnail_big:
            util.logger.warning(
                f"No thumbnail available for photo {self.photo.image_hash}"
            )
            return False

        util.logger.info("Generating captions with Im2txt")

        try:
            image_path = self.photo.thumbnail.thumbnail_big.path
            file_ext = str('.' + image_path.lower().split('.')[-1])
        except Exception:
            util.logger.warning(
                f"Cannot access thumbnail path for photo {self.photo.image_hash}"
            )
            return False
        if self.captions_json is None:
            self.captions_json = {}
        captions = self.captions_json

        try:
            caption, tag = generate_image_caption(image_path, file_ext)

            captions["im2txt"] = caption
            if tag:
                captions["im2txt_tag"] = tag
                self.captions_json = captions
                self._update_caption_generator_album_things(tag)
            else:
                self.captions_json = captions
            self.recreate_search_captions()
            if commit:
                self.save()

            util.logger.info(
                f"generated im2txt captions for image {image_path} with caption: {caption}, tag: {tag}"
            )
            return True
        except Exception:
            util.logger.exception(
                f"could not generate im2txt captions for image {image_path}"
            )
            return False

    def save_user_caption(self, caption, commit=True):
        """Save user-provided caption"""
        if not self.photo.thumbnail or not self.photo.thumbnail.thumbnail_big:
            util.logger.warning(
                f"No thumbnail available for photo {self.photo.image_hash}"
            )
            return False

        try:
            image_path = self.photo.thumbnail.thumbnail_big.path
        except Exception:
            util.logger.warning(
                f"Cannot access thumbnail path for photo {self.photo.image_hash}"
            )
            return False

        try:
            caption = caption.replace("<start>", "").replace("<end>", "").strip()

            if self.captions_json is None:
                self.captions_json = {}
            self.captions_json["user_caption"] = caption
            self.recreate_search_captions()

            if commit:
                self.save()

            util.logger.info(
                f"saved captions for image {image_path}. caption: {caption}. captions_json: {self.captions_json}."
            )

            # Handle hashtags
            hashtags = [
                word
                for word in caption.split()
                if word.startswith("#") and len(word) > 1
            ]

            for hashtag in hashtags:
                album_thing = api.models.album_thing.get_album_thing(
                    title=hashtag,
                    owner=self.photo.owner,
                    thing_type="hashtag_attribute",
                )
                if (
                    album_thing.photos.filter(image_hash=self.photo.image_hash).count()
                    == 0
                ):
                    album_thing.photos.add(self.photo)
                    album_thing.save()

            for album_thing in api.models.album_thing.AlbumThing.objects.filter(
                Q(photos__in=[self.photo])
                & Q(thing_type="hashtag_attribute")
                & Q(owner=self.photo.owner)
            ).all():
                if album_thing.title not in caption:
                    album_thing.photos.remove(self.photo)
                    album_thing.save()
            return True
        except Exception:
            util.logger.exception(f"could not save captions for image {image_path}")
            return False

    def recreate_search_captions(self):
        """Recreate search captions - directly access PhotoSearch model"""
        from api.models.photo_search import PhotoSearch

        search_instance, created = PhotoSearch.objects.get_or_create(photo=self.photo)
        search_instance.recreate_search_captions()
        search_instance.save()

    def generate_tag_captions(self, commit=True):
        """Generate tag captions using the caption-generator service.

        Tags are returned alongside captions from generate_image_caption()
        and stored under the 'tag' key in captions_json.
        """
        if not self.photo.thumbnail or not self.photo.thumbnail.thumbnail_big:
            return

        # Skip if this photo already has tags from the caption generator
        if (
            self.captions_json is not None
            and self.captions_json.get("tag") is not None
        ):
            return

        try:
            image_path = self.photo.thumbnail.thumbnail_big.path
            file_ext = str('.' + image_path.lower().split('.')[-1])
        except Exception:
            util.logger.warning(
                f"Cannot access thumbnail path for photo {self.photo.image_hash}"
            )
            return

        try:
            caption, tag = generate_image_caption(image_path, file_ext)

            if self.captions_json is None:
                self.captions_json = {}

            if caption:
                self.captions_json["im2txt"] = caption

            if tag:
                self.captions_json["im2txt_tag"] = tag
                self._update_caption_generator_album_things(tag)

            self.recreate_search_captions()

            if commit:
                self.save()
            util.logger.info(
                f"generated caption and tags for image {image_path}."
            )
        except Exception as e:
            util.logger.exception(
                f"could not generate tags for image "
                f"{self.photo.main_file.path if self.photo.main_file else 'no main file'}"
            )
            raise e

    # def _update_places365_album_things(self, res_places365):
    #     """Create/update AlbumThing entries for Places365 tags."""
    #     # Remove old album associations for this photo
    #     for album_thing in api.models.album_thing.AlbumThing.objects.filter(
    #         Q(photos__in=[self.photo])
    #         & (
    #             Q(thing_type="places365_attribute")
    #             | Q(thing_type="places365_category")
    #         )
    #         & Q(owner=self.photo.owner)
    #     ).all():
    #         album_thing.photos.remove(self.photo)
    #         album_thing.save()

    #     if "attributes" in res_places365:
    #         for attribute in res_places365["attributes"]:
    #             album_thing = api.models.album_thing.get_album_thing(
    #                 title=attribute,
    #                 owner=self.photo.owner,
    #                 thing_type="places365_attribute",
    #             )
    #             album_thing.photos.add(self.photo)
    #             album_thing.save()

    #     if "categories" in res_places365:
    #         for category in res_places365["categories"]:
    #             album_thing = api.models.album_thing.get_album_thing(
    #                 title=category,
    #                 owner=self.photo.owner,
    #                 thing_type="places365_category",
    #             )
    #             album_thing.photos.add(self.photo)
    #             album_thing.save()

    # def _update_siglip2_album_things(self, siglip2_result):
    #     """Create/update AlbumThing entries for SigLIP 2 tags."""
    #     tags = siglip2_result.get("tags", [])

    #     # Remove old siglip2 album associations for this photo
    #     for album_thing in api.models.album_thing.AlbumThing.objects.filter(
    #         Q(photos__in=[self.photo])
    #         & Q(thing_type="siglip2_tag")
    #         & Q(owner=self.photo.owner)
    #     ).all():
    #         album_thing.photos.remove(self.photo)
    #         album_thing.save()

    #     for tag in tags:
    #         album_thing = api.models.album_thing.get_album_thing(
    #             title=tag,
    #             owner=self.photo.owner,
    #             thing_type="siglip2_tag",
    #         )
    #         album_thing.photos.add(self.photo)
    #         album_thing.save()

    def _update_caption_generator_album_things(self, tag_result):
        """Create/update AlbumThing entries for caption-generator tags."""
        if isinstance(tag_result, list):
            tags = tag_result
        elif isinstance(tag_result, dict):
            tags = tag_result.get("tags", [])
        else:
            tags = []

        # Remove old caption_generator album associations for this photo
        for album_thing in api.models.album_thing.AlbumThing.objects.filter(
            Q(photos__in=[self.photo])
            & Q(thing_type="caption_generator_tag")
            & Q(owner=self.photo.owner)
        ).all():
            album_thing.photos.remove(self.photo)
            album_thing.save()

        for tag in tags:
            album_thing = api.models.album_thing.get_album_thing(
                title=tag,
                owner=self.photo.owner,
                thing_type="caption_generator_tag",
            )
            album_thing.photos.add(self.photo)
            album_thing.save()

    # Backward-compatible alias
    def generate_places365_captions(self, commit=True):
        return self.generate_tag_captions(commit=commit)
