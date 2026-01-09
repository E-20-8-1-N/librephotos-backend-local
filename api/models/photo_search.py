import os
from django.db import models

import api.models
from api import util

import torch
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # Register HEIF opener for Pillow

VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "google/paligemma2-10b-ft-docci-448")

SPECIAL_IMAGE_FILE_EXTENSIONS = ['.gif', '.apng', '.svg', '.heic', '.tiff', '.webp', '.avif', '.ico', '.icns']
RAW_IMAGE_FILE_EXTENSIONS = [
  '.dng','.rwz', '.cr2', '.nrw', '.eip', '.raf', '.erf', '.rw2', '.nef',
  '.arw', '.k25', '.srf', '.dcr', '.raw', '.crw', '.bay', '.3fr', '.cs1',
  '.mef', '.orf', '.ari', '.sr2', '.kdc', '.mos', '.mfw', '.fff', '.cr3',
  '.srw', '.rwl', '.j6i', '.kc2', '.x3f', '.mrw', '.iiq', '.pef', '.cxi', '.mdc'
]

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
    
    def image_format_convertor(image_path, file_ext):
        """
        Convert image file to supporting type.
        Returns a PIL.Image object or None if extraction fails.
        """

        if file_ext in ['.gif', '.apng']:
            try:
                with Image.open(image_path) as img:
                    img.seek(1)
                    return img.convert("RGB")
            except Exception as e:
                util.logger.error(f"Failed to extract frame from {file_ext} image ({image_path}): {e}")
                return None
        elif file_ext in ['.heic', '.tiff', '.webp', '.avif', '.ico', '.icns']:
            try:
                from pillow_heif import register_heif_opener

                register_heif_opener()
                with Image.open(image_path) as imgs:
                    return imgs.convert("RGB")
            except Exception as e:
                util.logger.error(f"Failed to convert {file_ext} image ({image_path}): {e}")
                return None
        elif file_ext in ['.svg']:
            try:
                import cairosvg
                from io import BytesIO

                png_data = cairosvg.svg2png(url=image_path)
                with Image.open(BytesIO(png_data)) as svg_img:
                    return svg_img.convert("RGB")
            except Exception as e:
                util.logger.error(f"Failed to convert {file_ext} image ({image_path}): {e}")
                return None
        elif file_ext in RAW_IMAGE_FILE_EXTENSIONS:
            try:
                import rawpy

                with rawpy.imread(image_path) as raw:
                    rgb = raw.postprocess()
                    return Image.fromarray(rgb)
            except Exception as e:
                util.logger.error(f"Failed to convert raw image file {file_ext} image ({image_path}): {e}")
                return None
        else:
            util.logger.warning(f"Unsupported file: {image_path}")
            return None

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
                    import gc
                    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
                    
                    caption_processor = PaliGemmaProcessor.from_pretrained(VLM_MODEL_NAME)
                    caption_model = PaliGemmaForConditionalGeneration.from_pretrained(VLM_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto").eval()

                    image_path = self.photo.thumbnail.thumbnail_big.path
                    file_ext = image_path.lower().split('.')[-1]
                    prompt = "" # Leaving the prompt blank for pre-trained models

                    try:
                        if file_ext in SPECIAL_IMAGE_FILE_EXTENSIONS + RAW_IMAGE_FILE_EXTENSIONS:
                            image = self.image_format_convertor(image_path, file_ext)
                            # Process the image
                            inputs = caption_processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(caption_model.device)
                            input_len = inputs["input_ids"].shape[-1]
                        else:
                            with Image.open(image_path).convert("RGB") as imgs:
                                inputs = caption_processor(text=prompt, images=imgs, return_tensors="pt").to(torch.bfloat16).to(caption_model.device)
                                input_len = inputs["input_ids"].shape[-1]

                        with torch.inference_mode():
                            # Generate the caption
                            generation = caption_model.generate(**inputs, max_new_tokens=100, do_sample=False)
                            generation = generation[0][input_len:]
                            # Decode the caption
                            caption = caption_processor.decode(generation, skip_special_tokens=True)
                            
                        
                            util.logger.info(f"Generated caption for {image_path}: '{caption}'")
                            search_captions += caption + " "

                            caption_data = self.photo.caption_instance.captions_json
                            caption_data["im2txt"] = caption
                            self.photo.caption_instance.captions_json = caption_data
                            self.photo.caption_instance.save()

                            # Free memory
                            del generation
                            gc.collect()
                    except Exception as e:
                        util.logger.error(f"Failed to generate caption for {image_path}: {e}")
                        return None

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
