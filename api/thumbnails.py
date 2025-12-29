import os
import subprocess

import pyvips
import requests
from django.conf import settings
from PIL import Image
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
    try:
        if is_raw(input_path):
            if "thumbnails_big" in output_path:
                json = {
                    "source": input_path,
                    "destination": complete_path,
                    "height": output_height,
                }
                response = requests.post(f"http://{BACKEND_HOST}:8003/", json=json).json()
                return response["thumbnail"]
            else:
                # only encode raw image in worse case, smaller thumbnails can get created from the big thumbnail instead
                big_thumbnail_path = os.path.join(
                    settings.MEDIA_ROOT, "thumbnails_big", hash + file_type
                )
                x = pyvips.Image.thumbnail(
                    big_thumbnail_path,
                    10000,
                    height=output_height,
                    size=pyvips.enums.Size.DOWN,
                )
                x.write_to_file(complete_path, Q=95)
            return complete_path
        else:
            x = pyvips.Image.thumbnail(
                input_path, 10000, height=output_height, size=pyvips.enums.Size.DOWN
            )
            x.write_to_file(complete_path, Q=95)
            return complete_path
    except Exception as e:
        try:
            util.logger.warning(f"Pyvips failed for {input_path}, trying Pillow fallback. Error: {e}")
            with Image.open(input_path) as img:
                aspect_ratio = img.width / img.height
                new_width = int(output_height * aspect_ratio)
                img.thumbnail((new_width, output_height), Image.Resampling.LANCZOS)
                img.save(complete_path, quality=95)
            return complete_path
        except Exception as e_fallback:
            util.logger.error(f"Could not create thumbnail for file {input_path} using fallback. Error: {e_fallback}")
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
