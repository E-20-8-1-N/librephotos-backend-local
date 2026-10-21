"""External captions adapt to the upstream model, indexing and feature gates."""

from unittest.mock import Mock, patch

import requests
from constance.test import override_config
from django.test import SimpleTestCase, TestCase, override_settings

from api.classification import caption_generator
from api.classification.captioning import generate_external_caption
from api.models import PhotoCaption
from api.models.album_thing import AlbumThing
from api.tests.utils import create_test_photo, create_test_user


class CaptionGeneratorClientTest(SimpleTestCase):
    def test_embedded_sections_and_separate_field_fallback(self) -> None:
        self.assertEqual(
            ("A cat", "cat, sofa, Hello"),
            caption_generator.parse_reply(
                {"caption": "A cat\nobjects: cat, sofa\ntexts: Hello"}
            ),
        )
        self.assertEqual(
            ("A cat", "cat, Hello"),
            caption_generator.parse_reply(
                {"caption": "A cat\ntexts: Hello", "objects": "cat"}
            ),
        )

    @override_settings(CAPTION_GENERATOR_RETRIES=1, CAPTION_GENERATOR_RETRY_BACKOFF=0.1)
    def test_gateway_timeout_retries_then_returns_caption_and_tags(self) -> None:
        failed = requests.Response()
        failed.status_code = 504
        answer = Mock(spec=requests.Response)
        answer.json.return_value = {"caption": "A cat", "objects": "cat"}
        with (
            patch.object(caption_generator, "ensure_ready", return_value=True),
            patch.object(
                caption_generator.requests, "post", side_effect=[failed, answer]
            ) as post,
            patch.object(caption_generator.time, "sleep") as sleep,
        ):
            self.assertEqual(
                ("A cat", "cat"), caption_generator.generate("/photo.webp", ".webp")
            )
        self.assertEqual(2, post.call_count)
        self.assertEqual(
            {"file_path": "/photo.webp", "file_ext": ".webp"},
            post.call_args.kwargs["json"],
        )
        sleep.assert_called_once_with(0.1)

    @override_settings(CAPTION_GENERATOR_RETRIES=0)
    def test_exhausted_retry_does_not_sleep_or_return_empty_data(self) -> None:
        with (
            patch.object(caption_generator, "ensure_ready", return_value=True),
            patch.object(
                caption_generator.requests, "post", side_effect=requests.Timeout
            ),
            patch.object(caption_generator.time, "sleep") as sleep,
            self.assertRaises(requests.Timeout),
        ):
            caption_generator.generate("/photo.webp", ".webp")
        sleep.assert_not_called()

    def test_healthy_service_does_not_require_docker(self) -> None:
        with (
            patch.object(caption_generator, "_healthy", return_value=True),
            patch("docker.from_env") as docker,
        ):
            self.assertTrue(caption_generator.ensure_ready())
        docker.assert_not_called()

    def test_starts_existing_container_and_waits_for_health(self) -> None:
        with (
            patch.object(caption_generator, "_healthy", side_effect=[False, True]),
            patch("docker.from_env") as docker,
        ):
            container = docker.return_value.containers.get.return_value
            container.status = "exited"
            self.assertTrue(caption_generator.ensure_ready())
        container.start.assert_called_once()


class ExternalCaptionIntegrationTest(TestCase):
    def setUp(self) -> None:
        self.user = create_test_user()
        self.photo = create_test_photo(owner=self.user)
        self.caption, _ = PhotoCaption.objects.get_or_create(photo=self.photo)

    def test_one_reply_updates_search_and_owner_scoped_things(self) -> None:
        self.caption.captions_json = {
            "user_caption": "My own caption",
            "openclip_vitb32": {"tags": ["beach"]},
        }
        self.caption.save()
        with (
            patch.object(
                PhotoCaption, "resolve_thumbnail_path", return_value="/photo.webp"
            ),
            patch.object(
                caption_generator, "generate", return_value=("A cat", "cat, sofa")
            ),
        ):
            self.assertTrue(generate_external_caption(self.caption))
        self.caption.refresh_from_db()
        self.assertEqual("My own caption", self.caption.captions_json["user_caption"])
        self.assertEqual("A cat", self.caption.captions_json["im2txt"])
        self.assertEqual("cat, sofa", self.caption.captions_json["im2txt_tag"])
        self.assertIn("beach", self.photo.search_instance.search_captions)
        self.assertIn("A cat", self.photo.search_instance.search_captions)
        things = AlbumThing.objects.owned_by(self.user).filter(
            photos=self.photo, thing_type="caption_generator_tag"
        )
        self.assertEqual({"cat", "sofa"}, set(things.values_list("title", flat=True)))

    @override_settings(FEATURE_SCENE_CLASSIFICATION=False)
    def test_disabled_things_are_not_stored(self) -> None:
        with (
            patch.object(
                PhotoCaption, "resolve_thumbnail_path", return_value="/photo.webp"
            ),
            patch.object(caption_generator, "generate", return_value=("A cat", "cat")),
        ):
            generate_external_caption(self.caption)
        self.assertNotIn("im2txt_tag", self.caption.captions_json)
        self.assertFalse(AlbumThing.objects.filter(photos=self.photo).exists())

    @override_settings(FEATURE_IMAGE_CAPTIONING=False)
    def test_disabled_caption_does_not_overwrite_existing_caption(self) -> None:
        self.caption.captions_json = {"im2txt": "Existing caption"}
        self.caption.save()
        with (
            patch.object(
                PhotoCaption, "resolve_thumbnail_path", return_value="/photo.webp"
            ),
            patch.object(
                caption_generator, "generate", return_value=("New caption", "cat")
            ),
        ):
            generate_external_caption(self.caption)
        self.assertEqual("Existing caption", self.caption.captions_json["im2txt"])
        self.assertEqual("cat", self.caption.captions_json["im2txt_tag"])

    @override_config(CAPTIONING_MODEL="lfm2_vl_450m")
    def test_external_model_does_not_download_builtin_caption_weights(self) -> None:
        from api.runtime.ml_models import MlTypes, _is_model_selected

        self.assertFalse(
            _is_model_selected(
                {"type": MlTypes.CAPTIONING, "name": "obsolete-captioner"}
            )
        )
