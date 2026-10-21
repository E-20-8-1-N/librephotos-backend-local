"""Manual generation uses the external container without a local model download."""

from unittest.mock import patch

from constance.test import override_config
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from api.models import PhotoCaption
from api.tests.utils import api_error, create_test_photo, create_test_user


class PhotoCaptionsTest(TestCase):
    def setUp(self) -> None:
        self.client = APIClient()
        self.user = create_test_user()
        self.client.force_authenticate(self.user)
        self.photo = create_test_photo(owner=self.user)

    @override_config(CAPTIONING_MODEL="None")
    def test_none_model_still_generates_caption_and_things_without_a_download(
        self,
    ) -> None:
        with (
            patch(
                "api.classification.caption_generator.generate",
                return_value=("A cat", "cat, sofa"),
            ) as generate,
            patch("api.runtime.ml_models.start_model_download") as download,
            patch("api.metadata.photo_writer.write_metadata") as write,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.post(
                "/api/photosedit/generateim2txt/",
                {"image_hash": self.photo.image_hash},
                format="json",
            )
        self.assertEqual(200, response.status_code)
        self.assertTrue(response.data["status"])
        generate.assert_called_once()
        write.assert_not_called()
        download.assert_not_called()
        captions = PhotoCaption.objects.get(photo=self.photo).captions_json
        self.assertEqual("A cat", captions["im2txt"])
        self.assertEqual("cat, sofa", captions["im2txt_tag"])

    def test_manual_action_regenerates_existing_outputs(self) -> None:
        caption, _ = PhotoCaption.objects.get_or_create(photo=self.photo)
        caption.captions_json = {"im2txt": "Old", "im2txt_tag": "old"}
        caption.save()
        with patch(
            "api.classification.caption_generator.generate", return_value=("New", "new")
        ) as generate:
            response = self.client.post(
                "/api/photosedit/generateim2txt/",
                {"image_hash": self.photo.image_hash},
                format="json",
            )
        self.assertEqual(200, response.status_code)
        generate.assert_called_once()
        caption.refresh_from_db()
        self.assertEqual("New", caption.captions_json["im2txt"])

    def test_generator_failure_is_reported(self) -> None:
        with patch(
            "api.classification.caption_generator.generate",
            side_effect=RuntimeError("container unavailable"),
        ):
            response = self.client.post(
                "/api/photosedit/generateim2txt/",
                {"image_hash": self.photo.image_hash},
                format="json",
            )
        self.assertEqual(500, response.status_code)
        self.assertEqual("caption_failed", api_error(response)["code"])

    @override_settings(FEATURE_IMAGE_CAPTIONING=False)
    def test_disabled_feature_does_not_call_the_container(self) -> None:
        with patch("api.classification.caption_generator.generate") as generate:
            response = self.client.post(
                "/api/photosedit/generateim2txt/",
                {"image_hash": self.photo.image_hash},
                format="json",
            )
        self.assertEqual(403, response.status_code)
        generate.assert_not_called()

    def test_another_users_photo_cannot_be_generated(self) -> None:
        other = create_test_photo(owner=create_test_user())
        with patch("api.classification.caption_generator.generate") as generate:
            response = self.client.post(
                "/api/photosedit/generateim2txt/",
                {"image_hash": other.image_hash},
                format="json",
            )
        self.assertEqual(404, response.status_code)
        generate.assert_not_called()
