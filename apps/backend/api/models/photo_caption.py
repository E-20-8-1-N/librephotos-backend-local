import os

from django.conf import settings
from django.db import models
from django.db.models import Q

import api.models
from api import util
import requests
import gc
import torch
import time
from api.image_captioning import generate_caption
from api.llm import generate_prompt
from api.models.user import User

BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")
CAPTION_GENERATOR_HOST = os.getenv("CAPTION_GENERATOR_HOST", "caption-generator")
CAPTION_GENERATOR_PORT = int(os.getenv("CAPTION_GENERATOR_PORT", 8020))
CAPTION_GENERATOR_API_ENDPOINT = os.getenv("CAPTION_GENERATOR_API_ENDPOINT", "generate")
CAPTION_GENERATOR_TIMEOUT_SEC = int(os.getenv("CAPTION_GENERATOR_TIMEOUT_SEC", 300))
CAPTION_GENERATOR_RETRIES = int(os.getenv("CAPTION_GENERATOR_RETRIES", 5))
CAPTION_GENERATOR_RETRY_BACKOFF = float(os.getenv("CAPTION_GENERATOR_RETRY_BACKOFF", 2.0))

def generate_image_caption(image_path: str, file_ext: str):
    """
    Generates a caption by sending the image to the caption-generator via HTTP request.
    """
    CAPTION_GENERATOR_API_URL = f"http://{CAPTION_GENERATOR_HOST}:{CAPTION_GENERATOR_PORT}/{CAPTION_GENERATOR_API_ENDPOINT}"

    try:
        payload = {"file_path": image_path, "file_ext": file_ext}

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
                    caption = result.get("caption", "").strip()
                    if caption:
                        util.logger.info(f"Generated caption for {image_path}: '{caption}'")
                        return caption
                    util.logger.error("Caption API returned empty caption for %s", image_path)
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
    except Exception as e:
        util.logger.error(f"Failed to generate caption for {image_path}: {e}", exc_info=True)
        pass
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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

    def _resolve_thumbnail_path(self):
        """Path of the big thumbnail, or None when it is missing or unreadable"""
        if not self.photo.thumbnail or not self.photo.thumbnail.thumbnail_big:
            util.logger.warning(
                f"No thumbnail available for photo {self.photo.image_hash}"
            )
            return None

        try:
            return self.photo.thumbnail.thumbnail_big.path
        except Exception:
            util.logger.warning(
                f"Cannot access thumbnail path for photo {self.photo.image_hash}"
            )
            return None

    def _llm_caption_context(self, llm_settings):
        """(person name, location, keywords flag) for prompting, or None if off"""
        from constance import config as site_config

        if str(site_config.LLM_MODEL).lower() == "none" or not llm_settings["enabled"]:
            return None

        face = api.models.Face.objects.filter(photo=self.photo).first()
        person_name = face.person.name if face and llm_settings["add_person"] else None

        search_instance = self.photo.search_instance
        location = None
        if (
            search_instance
            and search_instance.search_location
            and llm_settings["add_location"]
        ):
            location = search_instance.search_location

        return person_name, location, llm_settings["add_keywords"]

    @staticmethod
    def _im2txt_llm_prompt(caption, context):
        person_name, location, add_keywords = context
        person = f" Person: {person_name}" if person_name is not None else ""
        place = f" Place: {location}" if location is not None else ""
        keywords = " and tags or keywords" if add_keywords else ""
        return (
            "Q: Your task is to improve the following image caption: "
            + caption
            + ". You also know the following information about the image:"
            + place
            + person
            + ". Stick as closely as possible to the caption, while replacing generic information with information you know about the image. Only output the caption"
            + keywords
            + ". \n A:"
        )

    @staticmethod
    def _moondream_prompt(context):
        if context is None:
            return "Describe this image in a short, natural image caption."

        person_name, location, add_keywords = context
        person = ""
        if person_name is not None:
            person = (
                f" The person in the photo is named {person_name}. "
                f"Use the name '{person_name}' directly in the caption — do not say 'a person named'. "
                f"Keep the caption casual and to the point, like a friend tagging a photo."
            )
        place = f" This photo was taken at {location}." if location is not None else ""
        keywords = " Include relevant tags and keywords." if add_keywords else ""
        return "Write a short, natural image caption." + person + place + keywords

    def _store_generated_caption(self, captions, caption, commit):
        captions["im2txt"] = caption
        self.captions_json = captions
        self.recreate_search_captions()
        if commit:
            self.save()

    def generate_captions_im2txt(self, commit=True):
        """Generate im2txt captions for the photo"""
        if not settings.FEATURE_IMAGE_CAPTIONING:
            util.logger.info("Image captioning is disabled")
            return False

        util.logger.info("Generating captions with Im2txt")

        image_path = self._resolve_thumbnail_path()
        if image_path is None:
            return False
        file_ext = os.path.splitext(image_path)[1].lower()

        if self.captions_json is None:
            self.captions_json = {}
        captions = self.captions_json

        try:
            caption = generate_image_caption(image_path, file_ext)
            self._store_generated_caption(captions, caption, commit)

            util.logger.info(
                f"generated im2txt captions for image {image_path} with caption: {caption}"
            )
            return True
        except Exception:
            util.logger.exception(
                f"could not generate im2txt captions for image {image_path}"
            )
            return False

    def _generate_captions_moondream(self, commit=True):
        """Generate captions using Moondream with enhanced prompt"""
        image_path = self._resolve_thumbnail_path()
        if image_path is None:
            return False

        if self.captions_json is None:
            self.captions_json = {}
        captions = self.captions_json

        try:
            util.logger.info("Generating Moondream captions")

            llm_settings = User.objects.get(username=self.photo.owner).llm_settings
            prompt = self._moondream_prompt(self._llm_caption_context(llm_settings))
            util.logger.info(f"Moondream prompt: {prompt}")

            caption = generate_prompt(image_path=image_path, prompt=prompt)
            caption = caption.replace("<start>", "").replace("<end>", "").strip()

            self._store_generated_caption(captions, caption, commit)

            util.logger.info(
                f"Generated Moondream captions for image {image_path}, caption: {caption}"
            )
            return True
        except Exception:
            util.logger.exception(
                f"Could not generate Moondream captions for image {image_path}"
            )
            return False

    def save_user_caption(self, caption, commit=True):
        """Save user-provided caption"""
        image_path = self._resolve_thumbnail_path()
        if image_path is None:
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

            self._sync_hashtag_album_things(caption)
            return True
        except Exception:
            util.logger.exception(f"could not save captions for image {image_path}")
            return False

    def _photo_album_things(self, thing_types):
        """AlbumThings of the given types that own this photo, for this owner"""
        return api.models.album_thing.AlbumThing.objects.filter(
            Q(photos__in=[self.photo])
            & Q(thing_type__in=thing_types)
            & Q(owner=self.photo.owner)
        ).all()

    def _detach_photo_from_album_things(self, thing_types):
        for album_thing in self._photo_album_things(thing_types):
            album_thing.photos.remove(self.photo)
            album_thing.save()

    def _attach_photo_to_album_things(self, titles, thing_type):
        for title in titles:
            album_thing = api.models.album_thing.get_album_thing(
                title=title,
                owner=self.photo.owner,
                thing_type=thing_type,
            )
            album_thing.photos.add(self.photo)
            album_thing.save()

    def _sync_hashtag_album_things(self, caption):
        """Add album things for hashtags in the caption, drop the ones gone from it"""
        hashtags = [
            word for word in caption.split() if word.startswith("#") and len(word) > 1
        ]

        for hashtag in hashtags:
            album_thing = api.models.album_thing.get_album_thing(
                title=hashtag,
                owner=self.photo.owner,
                thing_type="hashtag_attribute",
            )
            if album_thing.photos.filter(image_hash=self.photo.image_hash).count() == 0:
                album_thing.photos.add(self.photo)
                album_thing.save()

        for album_thing in self._photo_album_things(["hashtag_attribute"]):
            if album_thing.title not in caption:
                album_thing.photos.remove(self.photo)
                album_thing.save()

    def recreate_search_captions(self):
        """Recreate search captions - directly access PhotoSearch model"""
        from api.models.photo_search import PhotoSearch

        search_instance, created = PhotoSearch.objects.get_or_create(photo=self.photo)
        search_instance.recreate_search_captions()
        search_instance.save()

    def generate_tag_captions(self, commit=True):
        """Generate tag captions using the active tagging model (Places365 or SigLIP 2).

        Tags are stored per-model in captions_json and are never deleted when
        switching models -- only the active model's tags are generated / visible.
        """
        if not settings.FEATURE_SCENE_CLASSIFICATION:
            util.logger.info("Scene classification is disabled")
            return

        from constance import config as site_config

        tagging_model = site_config.TAGGING_MODEL

        if not self.photo.thumbnail or not self.photo.thumbnail.thumbnail_big:
            return

        # Skip if this photo already has tags from the active model
        if (
            self.captions_json is not None
            and self.captions_json.get(tagging_model) is not None
        ):
            return

        try:
            import requests

            from api.http_timeouts import TAGS

            image_path = self.photo.thumbnail.thumbnail_big.path
            confidence = self.photo.owner.confidence
            json_data = {
                "image_path": image_path,
                "confidence": confidence,
                "tagging_model": tagging_model,
            }
            response = requests.post(
                f"http://{BACKEND_HOST}:8011/generate-tags",
                json=json_data,
                timeout=TAGS,
            )

            if not response.ok:
                util.logger.warning(
                    f"Tag service returned status {response.status_code} "
                    f"for image {image_path}"
                )
                return

            try:
                response_json = response.json()
            except (ValueError, RuntimeError):
                util.logger.warning(
                    f"Tag service returned non-JSON response for image {image_path}"
                )
                return

            tags_result = response_json.get("tags")

            if tags_result is None:
                return
            if self.captions_json is None:
                self.captions_json = {}

            # Store under the model-specific key
            self.captions_json[tagging_model] = tags_result
            self.recreate_search_captions()

            if tagging_model == "siglip2":
                self._update_siglip2_album_things(tags_result)
            else:
                self._update_places365_album_things(tags_result)

            if commit:
                self.save()
            util.logger.info(f"generated {tagging_model} tags for image {image_path}.")
        except Exception as e:
            util.logger.exception(
                f"could not generate tags for image "
                f"{self.photo.main_file.path if self.photo.main_file else 'no main file'}"
            )
            raise e

    def _update_places365_album_things(self, res_places365):
        """Create/update AlbumThing entries for Places365 tags."""
        self._detach_photo_from_album_things(
            ["places365_attribute", "places365_category"]
        )

        if "attributes" in res_places365:
            self._attach_photo_to_album_things(
                res_places365["attributes"], "places365_attribute"
            )

        if "categories" in res_places365:
            self._attach_photo_to_album_things(
                res_places365["categories"], "places365_category"
            )

    def _update_siglip2_album_things(self, siglip2_result):
        """Create/update AlbumThing entries for SigLIP 2 tags."""
        tags = siglip2_result.get("tags", [])

        self._detach_photo_from_album_things(["siglip2_tag"])
        self._attach_photo_to_album_things(tags, "siglip2_tag")

    # Backward-compatible alias
    def generate_places365_captions(self, commit=True):
        return self.generate_tag_captions(commit=commit)
