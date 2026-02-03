import os
from django.db import models

import api.models
from api import util
import requests

import gc
import torch

CAPTION_GENERATOR_HOST = os.getenv("CAPTION_GENERATOR_HOST", "caption-generator")
CAPTION_GENERATOR_PORT = int(os.getenv("CAPTION_GENERATOR_PORT", 8020))
CAPTION_GENERATOR_API_ENDPOINT = os.getenv("CAPTION_GENERATOR_API_ENDPOINT", "generate")

class PhotoSearch(models.Model):
    """Model for handling photo search functionality"""

    photo = models.OneToOneField(
        "Photo",
        on_delete=models.CASCADE,
        related_name="search_instance",
        primary_key=True,
    )
    search_captions = models.TextField(blank=True, null=True, db_index=True)
    search_location = models.TextField(blank=True, null=True, db_index=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "api_photo_search"

    def __str__(self):
        return f"Search data for {self.photo.image_hash}"

    def recreate_search_captions(self):
        """Recreate search captions from all caption sources"""
        search_captions = ""

        # Get captions from the PhotoCaption model
        if hasattr(self.photo, "caption_instance") and self.photo.caption_instance:
            captions_json = self.photo.caption_instance.captions_json
            if captions_json:
                places365_captions = captions_json.get("places365", {})

                attributes = places365_captions.get("attributes", [])
                search_captions += " ".join(attributes) + " "

                categories = places365_captions.get("categories", [])
                search_captions += " ".join(categories) + " "

                environment = places365_captions.get("environment", "")
                search_captions += environment + " "

                user_caption = captions_json.get("user_caption", "")
                if user_caption:
                    search_captions += user_caption + " "

                im2txt_caption = captions_json.get("im2txt", "")
                if im2txt_caption:
                    search_captions += im2txt_caption + " "
                else:
                    CAPTION_GENERATOR_API_URL = f"http://{CAPTION_GENERATOR_HOST}:{CAPTION_GENERATOR_PORT}/{CAPTION_GENERATOR_API_ENDPOINT}"
                    try:
                        image_path = self.photo.thumbnail.thumbnail_big.path
                        file_ext = str('.' + image_path.lower().split('.')[-1])
                        payload = { 
                            "file_path": image_path, 
                            "file_ext": file_ext 
                        }
                        util.logger.info(f"Sending caption request to {CAPTION_GENERATOR_API_URL}")
                        response = requests.post(CAPTION_GENERATOR_API_URL, json=payload, timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            caption = result['caption'].strip()
                            util.logger.info(f"Generated caption for {image_path}: '{caption}'")
                        else:
                            try:
                                err_msg = response.json()
                            except:
                                err_msg = response.text
                            util.logger.error(f"API Error {response.status_code}: {err_msg}")
                        
                        # Save back to captions_json
                        caption_data = self.photo.caption_instance.captions_json
                        caption_data["im2txt"] = caption
                        self.photo.caption_instance.captions_json = caption_data
                        self.photo.caption_instance.save()

                        search_captions += caption + " "

                    except Exception as e:
                        util.logger.error(f"Failed to generate caption for {image_path}: {e}", exc_info=True)
                        pass

                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()


        # Add face/person names
        for face in api.models.face.Face.objects.filter(photo=self.photo).all():
            if face.person:
                search_captions += face.person.name + " "

        # Add file paths
        for file in self.photo.files.all():
            search_captions += file.path + " "

        # Add media type
        if self.photo.video:
            search_captions += "type: video "

        # Add camera and lens info
        if self.photo.camera:
            search_captions += self.photo.camera + " "

        if self.photo.lens:
            search_captions += self.photo.lens + " "

        self.search_captions = search_captions.strip()

        util.logger.debug(
            f"Recreated search captions for image {self.photo.image_hash}."
        )

    def update_search_location(self, geolocation_json):
        """Update search location from geolocation data"""
        if geolocation_json and "address" in geolocation_json:
            self.search_location = geolocation_json["address"]
        elif geolocation_json and "features" in geolocation_json:
            # Handle features format used in tests
            features = geolocation_json["features"]
            location_parts = [
                feature.get("text", "") for feature in features if feature.get("text")
            ]
            self.search_location = ", ".join(location_parts) if location_parts else ""
        else:
            self.search_location = ""

        util.logger.debug(
            f"Updated search location for image {self.photo.image_hash}: {self.search_location}"
        )
