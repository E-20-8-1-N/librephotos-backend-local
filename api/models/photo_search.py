from django.db import models

import api.models
from api import util

import os
import requests
from constance import config as site_config

# --- Configuration (from Environment Variables) ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")

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
                    if site_config.CAPTIONING_MODEL == "moondream":
                        # Use custom prompt if provided, otherwise use default caption prompt
                        if prompt is None:
                            prompt = "Describe this image in a short, concise caption."

                        json_data = {
                            "image_path": self.photo.image_path,
                            "prompt": prompt,
                            "max_tokens": 256,
                        }
                        try:
                            response = requests.post(f"http://{BACKEND_HOST}:8008/generate", json=json_data)

                            if response.status_code != 201:
                                print(
                                    f"Error with Moondream captioning service: HTTP {response.status_code} - {response.text}"
                                )
                                return "Error generating caption with Moondream: Service unavailable"

                            response_data = response.json()
                            return response_data["response"]
                        except requests.exceptions.ConnectionError:
                            print(
                                "Error with Moondream captioning service: Cannot connect to LLM service on port 8008"
                            )
                            return "Error generating caption with Moondream: Service unavailable"
                        except requests.exceptions.Timeout:
                            print("Error with Moondream captioning service: Request timeout")
                            return "Error generating caption with Moondream: Request timeout"
                        except Exception as e:
                            print(f"Error with Moondream captioning service: {e}")
                            return "Error generating caption with Moondream"

                    blip = False
                    if site_config.CAPTIONING_MODEL == "blip_base_capfilt_large":
                        blip = True
                    # Original implementation for other models
                    json_data = {
                        "image_path": self.photo.image_path,
                        "onnx": False,
                        "blip": blip,
                    }
                    caption_response = requests.post(
                        f"http://{BACKEND_HOST}:8007/generate-caption", json=json_data
                    ).json()

                    search_captions += caption_response["caption"] + " " 

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
