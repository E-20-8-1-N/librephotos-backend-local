"""Generate captions and Things with the external caption-generator container."""

import logging
import os

from django.db import transaction

from api.classification import caption_generator
from api.models.photo_caption import PhotoCaption
from api.runtime.features import feature_enabled

logger = logging.getLogger(__name__)


def generate_photo_caption(caption: PhotoCaption, commit: bool = True) -> bool:
    """The manual caption action: regenerate caption and Things in the database.

    Captioning Model is a legacy setting; the container owns its model and no
    local model download is a prerequisite for this request.
    """
    if not feature_enabled("IMAGE_CAPTIONING"):
        return False
    try:
        return generate_external_caption(caption, commit=commit)
    except Exception:
        logger.exception(
            "Could not generate caption and Things for photo %s", caption.photo_id
        )
        return False


def generate_external_caption(
    caption: PhotoCaption, commit: bool = True, *, force: bool = True
) -> bool:
    """Store both outputs without writing to media files or sidecars.

    Automatic jobs use ``force=False`` to reuse complete outputs. The manual
    action explicitly regenerates. A reply is merged into the latest row so
    an edit made while inference ran is kept.
    """
    caption_on = feature_enabled("IMAGE_CAPTIONING")
    tags_on = feature_enabled("SCENE_CLASSIFICATION")
    if not caption_on and not tags_on:
        return False
    existing = caption.captions_json or {}
    infer = (
        force
        or (caption_on and not existing.get("im2txt"))
        or (tags_on and existing.get("im2txt_tag") is None)
    )
    if not infer:
        return True

    image_path = caption.resolve_thumbnail_path()
    if image_path is None:
        return False
    text, tags = caption_generator.generate(
        image_path, os.path.splitext(image_path)[1].lower()
    )
    text = text.replace("<start>", "").replace("<end>", "").strip()
    if caption_on and not text:
        raise RuntimeError("Caption generator returned no caption")

    if not commit:
        _apply_reply(caption, (text, tags), (caption_on, tags_on))
        return True

    with transaction.atomic():
        current = PhotoCaption.objects.select_for_update().get(pk=caption.pk)
        _apply_reply(current, (text, tags), (caption_on, tags_on))
        current.save(update_fields=["captions_json", "updated_at"])
        caption.captions_json = current.captions_json
    return True


def _apply_reply(
    caption: PhotoCaption, reply: tuple[str, str], enabled: tuple[bool, bool]
) -> None:
    captions = dict(caption.captions_json or {})
    if enabled[0]:
        captions["im2txt"] = reply[0]
    caption.captions_json = captions
    if enabled[1]:
        caption.store_caption_generator_tags(reply[1], commit=False)
    caption.recreate_search_captions()
