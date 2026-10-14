"""The external caption-generator client, shared by caption and Thing jobs."""

import logging
import time

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


def _url(endpoint: str) -> str:
    return (
        f"http://{settings.CAPTION_GENERATOR_HOST}:{settings.CAPTION_GENERATOR_PORT}/"
        f"{endpoint.lstrip('/')}"
    )


def _healthy() -> bool:
    try:
        response = requests.get(
            _url(settings.CAPTION_GENERATOR_HEALTH_ENDPOINT), timeout=5
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def ensure_ready() -> bool:
    """Check health, starting the existing Docker container when necessary."""
    if _healthy():
        return True
    try:
        import docker

        client = docker.from_env()
        try:
            container = client.containers.get(settings.CAPTION_GENERATOR_HOST)
            container.reload()
            if container.status != "running":
                container.start()
        finally:
            client.close()
    except Exception:
        logger.exception("Could not start the caption-generator container")
        return False

    deadline = time.monotonic() + settings.CAPTION_GENERATOR_STARTUP_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if _healthy():
            return True
        time.sleep(settings.CAPTION_GENERATOR_RETRY_BACKOFF)
    logger.error("Caption generator did not become healthy before the startup timeout")
    return False


def _text(body: dict[str, object], key: str) -> str:
    value = body.get(key)
    return value.strip() if isinstance(value, str) else ""


def parse_reply(body: dict[str, object]) -> tuple[str, str]:
    """Split embedded objects/texts, with separate JSON fields as fallbacks."""
    caption = _text(body, "caption")
    objects, texts = "", ""
    if "\nobjects:" in caption:
        caption, remainder = caption.split("\nobjects:", 1)
        objects, _, texts = remainder.partition("\ntexts:")
    elif "\ntexts:" in caption:
        caption, texts = caption.split("\ntexts:", 1)
    objects = objects.strip() or _text(body, "objects")
    texts = texts.strip() or _text(body, "texts")
    return caption.strip(), ", ".join(part for part in (objects, texts) if part)


def generate(image_path: str, file_ext: str) -> tuple[str, str]:
    """Generate a caption and Things in one request, with bounded retries."""
    if not ensure_ready():
        raise RuntimeError("Caption generator is unavailable")
    attempts = settings.CAPTION_GENERATOR_RETRIES + 1
    for attempt in range(attempts):
        try:
            response = requests.post(
                _url(settings.CAPTION_GENERATOR_API_ENDPOINT),
                json={"file_path": image_path, "file_ext": file_ext},
                timeout=settings.CAPTION_GENERATOR_TIMEOUT_SEC,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Caption generator returned a non-object response")
            caption, tags = parse_reply(body)
            if not caption and not tags:
                raise ValueError("Caption generator returned no caption or Things")
            return caption, tags
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as error:
            failed = getattr(error, "response", None)
            if attempt + 1 == attempts or (
                failed is not None and failed.status_code < 500
            ):
                raise
            delay = settings.CAPTION_GENERATOR_RETRY_BACKOFF * 2**attempt
            logger.warning("Caption request failed, retrying in %.1f seconds", delay)
            time.sleep(delay)
    raise RuntimeError("Caption generator exhausted its retries")
