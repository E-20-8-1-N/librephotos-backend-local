"""Tests for Pillow thumbnail rendering, orientation, and branch dispatch."""

import os
from unittest import mock

from django.test import SimpleTestCase, override_settings
from PIL import Image

from api.thumbnails import (
    _apply_local_orientation,
    _render_thumbnail,
    _reorient_file_in_place,
    _request_raw_thumbnail,
    _resize_big_thumbnail,
    create_thumbnail,
)

MEDIA_ROOT = os.path.join("/tmp", "lp-crap-u30-media")


def _make_image():
    """Return a 3x2 image whose six pixels are all distinct."""
    return Image.frombytes("L", (3, 2), bytes([0, 1, 2, 3, 4, 5]))


def _pixels(image):
    return list(image.tobytes())


class ApplyLocalOrientationTests(SimpleTestCase):
    def setUp(self):
        self.image = _make_image()
        self.addCleanup(self.image.close)

    def assert_transform(self, orientation, size, pixels):
        result = _apply_local_orientation(self.image, orientation)
        self.assertEqual(result.size, size)
        self.assertEqual(_pixels(result), pixels)

    def test_orientation_1_returns_same_object(self):
        self.assertIs(_apply_local_orientation(self.image, 1), self.image)

    def test_orientation_none_returns_same_object(self):
        self.assertIs(_apply_local_orientation(self.image, None), self.image)

    def test_orientation_2_flips_horizontally(self):
        self.assert_transform(2, (3, 2), [2, 1, 0, 5, 4, 3])

    def test_orientation_3_rotates_180(self):
        self.assert_transform(3, (3, 2), [5, 4, 3, 2, 1, 0])

    def test_orientation_4_flips_vertically(self):
        self.assert_transform(4, (3, 2), [3, 4, 5, 0, 1, 2])

    def test_orientation_5_transposes_top_left_to_bottom_right(self):
        self.assert_transform(5, (2, 3), [0, 3, 1, 4, 2, 5])

    def test_orientation_6_rotates_90_clockwise(self):
        self.assert_transform(6, (2, 3), [3, 0, 4, 1, 5, 2])

    def test_orientation_7_transposes_top_right_to_bottom_left(self):
        self.assert_transform(7, (2, 3), [5, 2, 4, 1, 3, 0])

    def test_orientation_8_rotates_90_counterclockwise(self):
        self.assert_transform(8, (2, 3), [2, 5, 1, 4, 0, 3])

    def test_unknown_orientations_return_image_unchanged(self):
        for value in (0, 9, -1, 42):
            with self.subTest(value=value):
                self.assertIs(_apply_local_orientation(self.image, value), self.image)

    def test_orientations_5_and_7_are_distinct(self):
        self.assertNotEqual(
            _pixels(_apply_local_orientation(self.image, 5)),
            _pixels(_apply_local_orientation(self.image, 7)),
        )


class PillowHelperTests(SimpleTestCase):
    def test_render_thumbnail_uses_pillow_pipeline(self):
        opened = mock.MagicMock(name="opened")
        opened.mode = "RGB"
        manager = mock.MagicMock()
        manager.__enter__.return_value = opened

        with mock.patch("api.thumbnails.Image.open", return_value=manager) as opener:
            with mock.patch(
                "api.thumbnails.ImageOps.exif_transpose", return_value=opened
            ) as exif_transpose:
                with mock.patch(
                    "api.thumbnails._apply_local_orientation"
                ) as apply_orientation:
                    result = _render_thumbnail(
                        "/data/photo.jpg", 200, "/media/thumb.webp", 1
                    )

        self.assertEqual(result, "/media/thumb.webp")
        opener.assert_called_once_with("/data/photo.jpg")
        opened.draft.assert_called_once_with("RGB", (800, 800))
        exif_transpose.assert_called_once_with(opened)
        opened.thumbnail.assert_called_once_with((10000, 200), Image.Resampling.LANCZOS)
        apply_orientation.assert_not_called()
        opened.save.assert_called_once_with(
            "/media/thumb.webp", quality=95, optimize=True
        )
        manager.__exit__.assert_called_once()

    def test_render_thumbnail_converts_and_applies_local_orientation(self):
        opened = mock.MagicMock(name="opened")
        upright = mock.MagicMock(name="upright")
        upright.mode = "RGBA"
        converted = upright.convert.return_value
        transformed = mock.MagicMock(name="transformed")
        manager = mock.MagicMock()
        manager.__enter__.return_value = opened

        with mock.patch("api.thumbnails.Image.open", return_value=manager):
            with mock.patch(
                "api.thumbnails.ImageOps.exif_transpose", return_value=upright
            ):
                with mock.patch(
                    "api.thumbnails._apply_local_orientation",
                    return_value=transformed,
                ) as apply_orientation:
                    _render_thumbnail("/data/photo.png", 100, "/media/thumb.webp", 6)

        upright.convert.assert_called_once_with("RGB")
        converted.thumbnail.assert_called_once_with(
            (10000, 100), Image.Resampling.LANCZOS
        )
        apply_orientation.assert_called_once_with(converted, 6)
        transformed.save.assert_called_once_with(
            "/media/thumb.webp", quality=95, optimize=True
        )
        converted.save.assert_not_called()

    def test_render_thumbnail_tolerates_decoder_without_draft_support(self):
        opened = mock.MagicMock(name="opened")
        opened.mode = "RGB"
        opened.draft.side_effect = RuntimeError("draft unsupported")
        manager = mock.MagicMock()
        manager.__enter__.return_value = opened

        with mock.patch("api.thumbnails.Image.open", return_value=manager):
            with mock.patch(
                "api.thumbnails.ImageOps.exif_transpose", return_value=opened
            ):
                _render_thumbnail("/data/photo.png", 50, "/media/thumb.webp", None)

        opened.thumbnail.assert_called_once_with((10000, 50), Image.Resampling.LANCZOS)
        opened.save.assert_called_once()

    def test_reorient_identity_does_not_open_file(self):
        with mock.patch("api.thumbnails.Image.open") as opener:
            _reorient_file_in_place("/media/raw.webp", 1)
            _reorient_file_in_place("/media/raw.webp", None)

        opener.assert_not_called()

    def test_reorient_applies_orientation_to_an_independent_copy(self):
        opened = mock.MagicMock(name="opened")
        upright = mock.MagicMock(name="upright")
        copied = upright.copy.return_value
        copied.mode = "RGB"
        transformed = mock.MagicMock(name="transformed")
        transformed.mode = "RGB"
        manager = mock.MagicMock()
        manager.__enter__.return_value = opened

        with mock.patch("api.thumbnails.Image.open", return_value=manager) as opener:
            with mock.patch(
                "api.thumbnails.ImageOps.exif_transpose", return_value=upright
            ) as exif_transpose:
                with mock.patch(
                    "api.thumbnails._apply_local_orientation",
                    return_value=transformed,
                ) as apply_orientation:
                    _reorient_file_in_place("/media/raw.webp", 8)

        opener.assert_called_once_with("/media/raw.webp")
        exif_transpose.assert_called_once_with(opened)
        upright.copy.assert_called_once_with()
        apply_orientation.assert_called_once_with(copied, 8)
        transformed.save.assert_called_once_with(
            "/media/raw.webp", quality=95, optimize=True
        )


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class RawThumbnailHelperTests(SimpleTestCase):
    def test_request_uses_backend_host_and_reorients_service_output(self):
        complete_path = os.path.join(MEDIA_ROOT, "thumbnails_big", "deadbeef.webp")
        response = mock.MagicMock()
        response.json.return_value = {"thumbnail": "/service/result.webp"}

        with mock.patch("api.thumbnails.BACKEND_HOST", "thumbnail-backend"):
            with mock.patch(
                "api.thumbnails.requests.post", return_value=response
            ) as post:
                with mock.patch("api.thumbnails._reorient_file_in_place") as reorient:
                    result = _request_raw_thumbnail(
                        "/data/photo.CR2", 800, complete_path, 8
                    )

        from api.http_timeouts import THUMBNAIL

        self.assertEqual(result, "/service/result.webp")
        post.assert_called_once_with(
            "http://thumbnail-backend:8003/",
            json={
                "source": "/data/photo.CR2",
                "destination": complete_path,
                "height": 800,
            },
            timeout=THUMBNAIL,
        )
        response.json.assert_called_once_with()
        reorient.assert_called_once_with(complete_path, 8)

    def test_missing_thumbnail_key_is_not_hidden(self):
        response = mock.MagicMock()
        response.json.return_value = {}
        with mock.patch("api.thumbnails.requests.post", return_value=response):
            with mock.patch("api.thumbnails._reorient_file_in_place"):
                with self.assertRaises(KeyError):
                    _request_raw_thumbnail("/data/photo.CR2", 800, "/media/raw.webp", 1)

    def test_resize_uses_the_already_oriented_big_thumbnail(self):
        complete_path = os.path.join(MEDIA_ROOT, "thumbnails_small", "deadbeef.webp")
        expected_big = os.path.join(MEDIA_ROOT, "thumbnails_big", "deadbeef.webp")
        with mock.patch(
            "api.thumbnails._render_thumbnail", return_value=complete_path
        ) as render:
            result = _resize_big_thumbnail(200, complete_path, "deadbeef", ".webp")

        self.assertEqual(result, complete_path)
        render.assert_called_once_with(expected_big, 200, complete_path, None)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class CreateThumbnailDispatchTests(SimpleTestCase):
    INPUT_PATH = "/data/photo.jpg"
    RAW_PATH = "/data/photo.CR2"

    def test_non_raw_dispatches_to_pillow_renderer(self):
        expected = os.path.join(MEDIA_ROOT, "thumbnails_big", "abc123.webp")
        with mock.patch("api.thumbnails.is_raw", return_value=False) as is_raw:
            with mock.patch(
                "api.thumbnails._render_thumbnail", return_value=expected
            ) as render:
                with mock.patch("api.thumbnails._request_raw_thumbnail") as request:
                    with mock.patch("api.thumbnails._resize_big_thumbnail") as resize:
                        result = create_thumbnail(
                            self.INPUT_PATH,
                            200,
                            "thumbnails_big",
                            "abc123",
                            ".webp",
                            local_orientation=6,
                        )

        self.assertEqual(result, expected)
        is_raw.assert_called_once_with(self.INPUT_PATH)
        render.assert_called_once_with(self.INPUT_PATH, 200, expected, 6)
        request.assert_not_called()
        resize.assert_not_called()

    def test_big_raw_dispatches_to_thumbnail_service(self):
        complete = os.path.join(MEDIA_ROOT, "thumbnails_big", "deadbeef.webp")
        with mock.patch("api.thumbnails.is_raw", return_value=True):
            with mock.patch(
                "api.thumbnails._request_raw_thumbnail",
                return_value="/service/result.webp",
            ) as request:
                with mock.patch("api.thumbnails._resize_big_thumbnail") as resize:
                    result = create_thumbnail(
                        self.RAW_PATH,
                        800,
                        "thumbnails_big",
                        "deadbeef",
                        ".webp",
                        local_orientation=8,
                    )

        self.assertEqual(result, "/service/result.webp")
        request.assert_called_once_with(self.RAW_PATH, 800, complete, 8)
        resize.assert_not_called()

    def test_small_raw_dispatches_to_big_thumbnail_resize(self):
        complete = os.path.join(MEDIA_ROOT, "thumbnails_small", "deadbeef.webp")
        with mock.patch("api.thumbnails.is_raw", return_value=True):
            with mock.patch("api.thumbnails._request_raw_thumbnail") as request:
                with mock.patch(
                    "api.thumbnails._resize_big_thumbnail", return_value=complete
                ) as resize:
                    result = create_thumbnail(
                        self.RAW_PATH,
                        200,
                        "thumbnails_small",
                        "deadbeef",
                        ".webp",
                        local_orientation=6,
                    )

        self.assertEqual(result, complete)
        request.assert_not_called()
        resize.assert_called_once_with(200, complete, "deadbeef", ".webp")

    def test_raw_big_substring_uses_thumbnail_service(self):
        complete = os.path.join(MEDIA_ROOT, "nested", "thumbnails_big", "x", "h.webp")
        with mock.patch("api.thumbnails.is_raw", return_value=True):
            with mock.patch(
                "api.thumbnails._request_raw_thumbnail", return_value="result"
            ) as request:
                create_thumbnail(
                    self.RAW_PATH, 800, "nested/thumbnails_big/x", "h", ".webp"
                )

        request.assert_called_once_with(self.RAW_PATH, 800, complete, 1)

    def test_exception_is_logged_with_context_and_reraised(self):
        boom = RuntimeError("Pillow exploded")
        with mock.patch("api.thumbnails.is_raw", return_value=False):
            with mock.patch("api.thumbnails._render_thumbnail", side_effect=boom):
                with mock.patch("api.thumbnails.util.logger") as logger:
                    with self.assertRaises(RuntimeError) as context:
                        create_thumbnail(
                            self.INPUT_PATH,
                            200,
                            "thumbnails_big",
                            "h",
                            ".webp",
                        )

        self.assertIs(context.exception, boom)
        logger.error.assert_called_once_with(
            f"Could not create thumbnail for file {self.INPUT_PATH}: Pillow exploded"
        )
