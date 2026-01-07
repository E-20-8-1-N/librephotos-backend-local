import os
import subprocess

import requests
from django.conf import settings
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from api import util
from api.models.file import is_raw

register_heif_opener()

BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")

_ORIENTATION_TRANSFORMS = {
    2: ImageOps.mirror,
    3: lambda image: image.rotate(180, expand=True),
    4: ImageOps.flip,
    5: lambda image: ImageOps.mirror(image.rotate(90, expand=True)),
    6: lambda image: image.rotate(-90, expand=True),
    7: lambda image: ImageOps.mirror(image.rotate(-90, expand=True)),
    8: lambda image: image.rotate(90, expand=True),
}


def _apply_local_orientation(
    image: Image.Image, local_orientation: int
) -> Image.Image:
    """Apply a user-specified orientation transform to an already-upright Pillow image.

    ``local_orientation`` follows the EXIF Orientation convention (1-8).
    Orientation 1 is the identity (no change).  The image passed in is assumed
    to be already auto-rotated by Pillow (i.e. it is visually upright), so
    this function applies *additional* rotation/flip on top.

    EXIF orientation semantics (applied to a visually-upright image):
        1 – no change
        2 – flip horizontal
        3 – rotate 180°
        4 – flip vertical
        5 – rotate 90° CCW then flip horizontal
        6 – rotate 90° CW
        7 – rotate 90° CW then flip horizontal
        8 – rotate 90° CCW (= 270° CW)
    """
    transform = _ORIENTATION_TRANSFORMS.get(local_orientation)
    if transform is None:
        return image
    return transform(image)


def _media_path(output_path, hash, file_type):
    return os.path.join(settings.MEDIA_ROOT, output_path, hash + file_type)


def _reorient_file_in_place(complete_path, local_orientation):
    if not local_orientation or local_orientation == 1:
        return
    with Image.open(complete_path) as image:
        image = ImageOps.exif_transpose(image).copy()
        image = _apply_local_orientation(image, local_orientation)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(complete_path, quality=95, optimize=True)


def _request_raw_thumbnail(input_path, output_height, complete_path, local_orientation):
    json = {
        "source": input_path,
        "destination": complete_path,
        "height": output_height,
    }
    from api.http_timeouts import THUMBNAIL

    response = requests.post(
        f"http://{BACKEND_HOST}:8003/", json=json, timeout=THUMBNAIL
    ).json()
    # The RAW service applies auto-orientation internally.  Apply
    # any user-specified rotation on top.
    _reorient_file_in_place(complete_path, local_orientation)
    return response["thumbnail"]


def _resize_big_thumbnail(output_height, complete_path, hash, file_type):
    big_thumbnail_path = os.path.join(
        settings.MEDIA_ROOT, "thumbnails_big", hash + file_type
    )
    # The big thumbnail already has EXIF auto-rotation and any
    # local_orientation applied, so we only resize here.
    return _render_thumbnail(big_thumbnail_path, output_height, complete_path, None)


def _render_thumbnail(input_path, output_height, complete_path, local_orientation):
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.thumbnail((10000, output_height), Image.Resampling.LANCZOS)
        if local_orientation and local_orientation != 1:
            image = _apply_local_orientation(image, local_orientation)
        image.save(complete_path, quality=95, optimize=True)
    return complete_path


def create_thumbnail(
    input_path, output_height, output_path, hash, file_type, local_orientation=1
):
    try:
        raw = is_raw(input_path)
        complete_path = _media_path(output_path, hash, file_type)
        if not raw:
            return _render_thumbnail(
                input_path, output_height, complete_path, local_orientation
            )
        if "thumbnails_big" in output_path:
            return _request_raw_thumbnail(
                input_path, output_height, complete_path, local_orientation
            )
        return _resize_big_thumbnail(output_height, complete_path, hash, file_type)
    except Exception as e:
        util.logger.error(f"Could not create thumbnail for file {input_path}: {e}")
        raise


def create_animated_thumbnail(input_path, output_height, output_path, hash, file_type):
    try:
        output = os.path.join(settings.MEDIA_ROOT, output_path, hash + file_type)
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-crf",
            "20",
            "-filter:v",
            f"scale=-2:{output_height}",
            output,
        ]

        with subprocess.Popen(command) as proc:
            proc.wait()
    except Exception as e:
        util.logger.error(f"Could not create animated thumbnail for file {input_path}")
        raise e


def create_thumbnail_for_video(input_path, output_path, hash, file_type):
    try:
        output = os.path.join(settings.MEDIA_ROOT, output_path, hash + file_type)
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            "00:00:00.000",
            "-vframes",
            "1",
            output,
        ]

        with subprocess.Popen(command) as proc:
            proc.wait()
    except Exception as e:
        util.logger.error(f"Could not create thumbnail for video file {input_path}")
        raise e


def does_static_thumbnail_exist(output_path, hash):
    return os.path.exists(
        os.path.join(settings.MEDIA_ROOT, output_path, hash + ".webp")
    )


def does_video_thumbnail_exist(output_path, hash):
    return os.path.exists(os.path.join(settings.MEDIA_ROOT, output_path, hash + ".mp4"))
