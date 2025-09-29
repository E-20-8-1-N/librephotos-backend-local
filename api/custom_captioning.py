import os
import logging
import requests
from api.models import Photo, PhotoCaption

logger = logging.getLogger(__name__)

# --- Configuration (from Environment Variables) ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")
AUTO_GENERATE_CAPTIONS_ON_SCAN = os.getenv("AUTO_GENERATE_CAPTIONS_ON_SCAN", True)
USE_EXTERNAL_CAPTION_SERVICE = os.getenv("USE_EXTERNAL_CAPTION_SERVICE", True)
CAPTION_SERVICE_URL = f"http://{BACKEND_HOST}:8007/generate-caption"
CAPTION_TIMEOUT = os.getenv("CAPTION_SERVICE_TIMEOUT", 30)


def generate_and_store_caption(photo: Photo, force: bool = False) -> bool:
    """
    Returns True if a caption was generated (or already existed and accepted), False on failure.
    """
    if not AUTO_GENERATE_CAPTIONS_ON_SCAN:
        return False

    # Get/create caption instance
    caption_instance, _ = PhotoCaption.objects.get_or_create(photo=photo)

    # If not forcing and we already have a machine_generated_caption (or however stored), skip
    # Adjust attribute names to your actual model fields.
    existing = getattr(caption_instance, "generated_caption", None) or getattr(
        caption_instance, "captions_json", None
    )
    if existing and not force:
        return True

    # Try internal method first if available
    if hasattr(caption_instance, "generate_captions_im2txt") and not USE_EXTERNAL_CAPTION_SERVICE:
        try:
            res = caption_instance.generate_captions_im2txt()
            if res:
                logger.info("Caption generated internally for photo %s", photo.image_hash)
                return True
            logger.warning("Internal caption generation returned falsy result for %s", photo.image_hash)
            return False
        except Exception:
            logger.exception("Internal caption generation failed for %s", photo.image_hash)
            return False

    if USE_EXTERNAL_CAPTION_SERVICE:
        try:
            payload = {"image_hash": photo.image_hash}
            resp = requests.post(CAPTION_SERVICE_URL, json=payload, timeout=CAPTION_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(
                    "External caption service failed for %s status=%s body=%s",
                    photo.image_hash,
                    resp.status_code,
                    resp.text[:300],
                )
                return False

            data = {}
            try:
                data = resp.json()
            except Exception:
                logger.warning("Caption service did not return JSON for %s", photo.image_hash)

            # Adapt to the actual response structure. Assume it returns {"caption": "..."}
            caption_text = data.get("caption")
            if not caption_text:
                logger.warning("No caption field in external response for %s", photo.image_hash)
                return False

            # Persist using existing save method if present
            if hasattr(caption_instance, "save_user_caption"):
                caption_instance.save_user_caption(caption_text)
            else:
                # Fallback direct assignment
                setattr(caption_instance, "generated_caption", caption_text)
                caption_instance.save(update_fields=["generated_caption"])
            logger.info("External caption stored for photo %s", photo.image_hash)
            return True

        except Exception:
            logger.exception("Exception calling external caption service for %s", photo.image_hash)
            return False

    return False