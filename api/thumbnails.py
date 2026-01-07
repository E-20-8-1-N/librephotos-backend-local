import os
import subprocess

import requests
from django.conf import settings
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
register_heif_opener() # Register HEIF opener for Pillow

from api import util
from api.models.file import is_raw

# --- Configuration (from Environment Variables) ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")

def _apply_local_orientation(image: Image.Image, local_orientation: int) -> Image.Image:
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
    if local_orientation == 1 or local_orientation is None:
        return image
    if local_orientation == 2:
        return ImageOps.mirror(image)
    if local_orientation == 3:
        return image.rotate(180, expand=True)
    if local_orientation == 4:
        return ImageOps.flip(image)
    if local_orientation == 5:
        # 90° CCW + flip H
        return ImageOps.mirror(image.rotate(90, expand=True))
    if local_orientation == 6:
        # pyvips.rot270() rotates 270° counter-clockwise, which is the same
        # as 90° clockwise — matching EXIF orientation 6 (display: 90° CW).
        return image.rotate(-90, expand=True) # or 270
    if local_orientation == 7:
        # 90° CW + flip H
        return ImageOps.mirror(image.rotate(-90, expand=True))
    if local_orientation == 8:
        # pyvips.rot90() rotates 90° counter-clockwise — matching EXIF
        # orientation 8 (display: 270° CW = 90° CCW).
        return image.rotate(90, expand=True)
    return image


def create_thumbnail(
    input_path, output_height, output_path, hash, file_type, local_orientation=1
):
    complete_path = os.path.join(
        settings.MEDIA_ROOT, output_path, hash + file_type
    )
    
    source_path = input_path

    try:
        # ====================== RAW File Handling ======================
        if is_raw(input_path):
            if "thumbnails_big" in output_path:
                json = {
                    "source": input_path,
                    "destination": complete_path,
                    "height": output_height,
                }
                response = requests.post(f"http://{BACKEND_HOST}:8003/",json=json).json()
                # The RAW service already applies auto-orientation.
                # Apply any additional user-specified local orientation on top.
                if local_orientation and local_orientation != 1:
                    with Image.open(complete_path) as img:
                        img = img.copy()  # Ensure we don't modify in-place unexpectedly
                        img = ImageOps.exif_transpose(img)  # Safety
                        img = _apply_local_orientation(img, local_orientation)
                        img = img.convert("RGB")
                        img.save(complete_path, quality=95, optimize=True)
                return response["thumbnail"]
            else:
                # Smaller thumbnails: derive from the already-created big thumbnail
                big_thumbnail_path = os.path.join(
                    settings.MEDIA_ROOT, "thumbnails_big", hash + file_type
                )
                source_path = big_thumbnail_path
        else:
            source_path = input_path
        # ====================== Pillow Processing ======================
        with Image.open(source_path) as img:
            # Apply EXIF-based auto-rotation (what pyvips did automatically)
            img = ImageOps.exif_transpose(img)
            # Convert to RGB if necessary (required for JPEG/WebP)
            if img.mode in ("RGBA", "P", "LA", "RGBA"):
                img = img.convert("RGB")
            # Resize while preserving aspect ratio (equivalent to pyvips thumbnail)
            img.thumbnail((10000, output_height), Image.Resampling.LANCZOS)
            # Apply additional user-specified orientation if needed
            if local_orientation and local_orientation != 1:
                img = _apply_local_orientation(img, local_orientation)
            # Save with high quality
            img.save(complete_path, quality=95, optimize=True)
        return complete_path
    except Exception as e:
        util.logger.error(f"Could not create thumbnail for file {input_path}: {e}")
        raise


def create_animated_thumbnail(input_path, output_height, output_path, hash, file_type):
    try:
        output = os.path.join(
            settings.MEDIA_ROOT, output_path, hash + file_type
        ).strip()
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
        output = os.path.join(
            settings.MEDIA_ROOT, output_path, hash + file_type
        ).strip()
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
        os.path.join(settings.MEDIA_ROOT, output_path, hash + ".webp").strip()
    )


def does_video_thumbnail_exist(output_path, hash):
    return os.path.exists(
        os.path.join(settings.MEDIA_ROOT, output_path, hash + ".mp4").strip()
    )