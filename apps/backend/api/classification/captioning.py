"""Captioning a photo with the image captioning sidecar."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import requests

from api.models.user import User
from api.runtime import sidecars
from api.runtime.features import feature_enabled
from api.runtime.http_timeouts import CAPTION

if TYPE_CHECKING:
    from api.models.photo_caption import PhotoCaption

logger = logging.getLogger(__name__)


class CaptionError(RuntimeError):
    """The captioning sidecar answered, but not with a caption."""


def generate_caption(image_path: str, prompt: str | None = None) -> str:
    """A caption for the photo from the image captioning sidecar.

    ``prompt`` steers the vision-language model; ``None`` asks for its plain
    one-sentence caption. Which model runs is the sidecar's business: there is
    one, and it is always available.

    Raises ``CaptionError`` carrying the sidecar's own error message when it
    could not caption the photo, so the reason (a missing model file, an
    ONNX Runtime failure) ends up in the backend log instead of a bare
    ``KeyError: 'caption'``. Connection and timeout errors propagate as-is.
    """
    json_data = {"image_path": image_path}
    if prompt is not None:
        json_data["prompt"] = prompt

    try:
        response = sidecars.post(
            "image_captioning", "/generate-caption", json=json_data, timeout=CAPTION
        )
    except requests.HTTPError as error:
        failed = error.response
        # sidecars.post raises only through raise_for_status(), which always
        # attaches the response.
        assert failed is not None
        raise CaptionError(
            f"captioning sidecar returned HTTP {failed.status_code}: {_detail(failed)}"
        ) from error

    body = _body(response)
    if "caption" not in body:
        raise CaptionError(
            f"captioning sidecar returned HTTP {response.status_code}: "
            f"{_detail(response)}"
        )
    return body["caption"]


def _body(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _detail(response: requests.Response) -> str:
    return (
        _body(response).get("error") or response.text.strip() or "no caption in reply"
    )


def generate_photo_caption(caption: PhotoCaption, commit: bool = True) -> bool:
    """Generate a caption for the photo with the captioning sidecar.

    The prompt carries the recognised person and the place when the user's
    caption settings allow it. Stored as ``captions_json["im2txt"]`` (the
    first model's name). Moved out of ``PhotoCaption`` so the model makes no
    sidecar calls (#2336).
    """
    if not feature_enabled("IMAGE_CAPTIONING"):
        logger.info("Image captioning is disabled")
        return False

    image_path = caption.resolve_thumbnail_path()
    if image_path is None:
        return False

    if caption.captions_json is None:
        caption.captions_json = {}
    captions = caption.captions_json

    try:
        from constance import config as site_config

        if str(site_config.CAPTIONING_MODEL).lower() == "none":
            logger.info("Generating captions is disabled")
            return False

        if uses_caption_generator():
            return generate_external_caption(caption, commit)

        llm_settings = User.objects.get(username=caption.photo.owner).llm_settings
        context = caption.caption_context(llm_settings)
        prompt = caption.caption_prompt(context)
        logger.debug("Caption prompt: %s", prompt)

        text = generate_caption(image_path=image_path, prompt=prompt)
        text = text.replace("<start>", "").replace("<end>", "").strip()

        caption.store_generated_caption(captions, text, commit)

        logger.info("generated caption for image %s: %s", image_path, text)
        return True
    except Exception:
        logger.exception("could not generate caption for image %s", image_path)
        return False


def uses_caption_generator() -> bool:
    """Whether captions and Things are supplied by the external generator."""
    from constance import config as site_config

    return site_config.CAPTIONING_MODEL == "caption_generator"


def generate_external_caption(caption: PhotoCaption, commit: bool = True) -> bool:
    """Store the caption and Things together, respecting both feature switches."""
    from api.classification import caption_generator

    image_path = caption.resolve_thumbnail_path()
    if image_path is None:
        return False
    text, tags = caption_generator.generate(
        image_path, os.path.splitext(image_path)[1].lower()
    )
    captions = dict(caption.captions_json or {})
    if text and feature_enabled("IMAGE_CAPTIONING"):
        captions["im2txt"] = text
    caption.captions_json = captions
    if feature_enabled("SCENE_CLASSIFICATION"):
        caption.store_caption_generator_tags(tags, commit=False)
    caption.recreate_search_captions()
    if commit:
        caption.save()
    return True
