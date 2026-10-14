"""Fork metadata normalization feeds the current upstream stores and rules."""

from unittest.mock import patch

from django.test import TestCase

from api.metadata.photo_reader import EXIF_VALUE_NAMES, extract_exif_data
from api.models import PhotoCaption
from api.tests.utils import create_test_photo, create_test_user


class NormalizedMetadataTest(TestCase):
    def test_normalized_values_populate_camera_fields_and_video_description(
        self,
    ) -> None:
        photo = create_test_photo(owner=create_test_user(), video=True)
        values = {
            "fstop": "f/2.8",
            "iso": ["ISO 400"],
            "focal_length": {"value": "50 mm"},
            "shutter_speed": "1/4000",
            "camera": "Camera",
            "lens": "Lens",
            "width": "1920",
            "height": "1080",
            "gps_latitude": "0",
            "gps_longitude": "1",
            "keys_description": {"x-default": "A video caption"},
            "xmp_title": {"en": "English", "x-default": "Default title"},
            "file_modify_date": "2025:09:03 17:51:06+08:00",
            "timezone_offset": "+08:00",
        }
        with patch(
            "api.metadata.photo_reader.get_metadata",
            return_value=[values.get(name) for name in EXIF_VALUE_NAMES],
        ):
            metadata = extract_exif_data(photo)
        self.assertIsNotNone(metadata)
        self.assertEqual(2.8, metadata.aperture)
        self.assertEqual(400, metadata.iso)
        self.assertEqual(50, metadata.focal_length)
        self.assertEqual("1/4000", metadata.shutter_speed)
        self.assertEqual("Default title", metadata.title)
        self.assertEqual("+08:00", metadata.timezone_offset)
        self.assertEqual(2025, metadata.date_modified.year)
        caption = PhotoCaption.objects.get(photo=photo)
        self.assertEqual("A video caption", caption.captions_json["user_caption"])
        self.assertEqual((0, 1), (photo.exif_gps_lat, photo.exif_gps_lon))
