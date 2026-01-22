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

def create_thumbnail(input_path, output_height, output_path, hash, file_type):
    complete_path = os.path.join(
        settings.MEDIA_ROOT, output_path, hash + file_type
    ).strip()
    
    source_path = input_path

    # Handle RAW files
    if is_raw(input_path):
        if "thumbnails_big" in output_path:
            json = {
                "source": input_path,
                "destination": complete_path,
                "height": output_height,
            }
            try:
                response = requests.post(f"http://{BACKEND_HOST}:8003/", json=json).json()
                return response["thumbnail"]
            except Exception as e:
                util.logger.error(f"Backend RAW processing failed for {input_path}: {e}")
                raise e
        else:
            source_path = os.path.join(
                settings.MEDIA_ROOT, "thumbnails_big", hash + file_type
            )
    # Process image using Pillow (for JPEGs, HEICs, PNGs, and pre-converted RAWs)
    try:
        with Image.open(source_path) as img:
            # Apply EXIF rotation (pyvips did this auto; Pillow needs explicit call)
            img = ImageOps.exif_transpose(img)
            
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                
            # Pillow's thumbnail method modifies the image in-place and preserves aspect ratio
            img.thumbnail((10000, output_height), Image.Resampling.LANCZOS)
            img.save(complete_path, quality=95)
            return complete_path
    except Exception as e:
        util.logger.error(f"Could not create thumbnail for file {input_path} using PIL: {e}")
        raise e


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
