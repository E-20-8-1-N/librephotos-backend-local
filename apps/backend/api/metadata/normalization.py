"""Normalize the fork's EXIF scalar, list and language-map inputs."""

import math
import re
from fractions import Fraction


def number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value if math.isfinite(value) else None
    if isinstance(value, list | dict):
        items = value.values() if isinstance(value, dict) else value
        return next(
            (normalized for item in items if (normalized := number(item)) is not None),
            None,
        )
    if not isinstance(value, str):
        return None
    try:
        return float(Fraction(value.strip()))
    except (ValueError, ZeroDivisionError):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(match.group()) if match else None


FLOAT_FIELDS = frozenset(
    {
        "fstop",
        "focal_length",
        "shutter_speed",
        "video_length",
        "matroska_duration",
        "gps_latitude",
        "gps_longitude",
        "gps_altitude",
        "subject_distance",
        "digital_zoom_ratio",
        "exposure_compensation",
    }
)
INT_FIELDS = frozenset(
    {
        "size",
        "iso",
        "width",
        "height",
        "focal_length_35mm",
        "rating",
        "image_number",
        "orientation",
        "bits_per_sample",
        "flash",
        "metering_mode",
        "white_balance",
    }
)


def file_values(values: dict[str, object]) -> dict[str, object]:
    """Normalize numbers without changing the upstream metadata stores/rules."""
    normalized = dict(values)
    for key in FLOAT_FIELDS | INT_FIELDS:
        if key in values:
            value = number(values[key])
            normalized[key] = (
                int(value) if value is not None and key in INT_FIELDS else value
            )
    return normalized
