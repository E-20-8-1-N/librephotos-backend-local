"""External generation stores both outputs without metadata write-back."""

from unittest.mock import PropertyMock, patch

from constance.test import override_config
from django.test import TestCase, override_settings

from api.classification.captioning import (
    generate_external_caption,
    generate_photo_caption,
)
from api.metadata import edits
from api.models import PhotoCaption, PhotoSearch, Thumbnail, User
from api.tests.utils import create_test_photo, create_test_user


class CaptionGenerationTest(TestCase):
    def setUp(self) -> None:
        self.user = create_test_user(save_metadata_to_disk=User.SaveMetadata.OFF)
        self.photo = create_test_photo(owner=self.user)
        self.caption, _ = PhotoCaption.objects.get_or_create(photo=self.photo)

    @override_config(CAPTIONING_MODEL="None")
    def test_none_model_uses_external_container_and_stores_both_outputs(self) -> None:
        with patch(
            "api.classification.caption_generator.generate",
            return_value=("<start> A cat <end>", "cat, sofa"),
        ) as generate:
            self.assertTrue(generate_photo_caption(self.caption))
        generate.assert_called_once()
        self.assertTrue(generate.call_args.args[0].endswith(".webp"))
        self.assertEqual(".webp", generate.call_args.args[1])
        self.caption.refresh_from_db()
        self.assertEqual("A cat", self.caption.captions_json["im2txt"])
        self.assertEqual("cat, sofa", self.caption.captions_json["im2txt_tag"])
        search = PhotoSearch.objects.get(photo=self.photo)
        self.assertIn("A cat", search.search_captions)
        self.assertIn("sofa", search.search_captions)

    @override_settings(FEATURE_IMAGE_CAPTIONING=False)
    def test_disabled_caption_feature_skips_manual_generation(self) -> None:
        with patch("api.classification.caption_generator.generate") as generate:
            self.assertFalse(generate_photo_caption(self.caption))
        generate.assert_not_called()

    def test_no_thumbnail_skips_the_container(self) -> None:
        Thumbnail.objects.filter(photo=self.photo).delete()
        caption = PhotoCaption.objects.get(pk=self.caption.pk)
        with patch("api.classification.caption_generator.generate") as generate:
            self.assertFalse(generate_photo_caption(caption))
        generate.assert_not_called()

    def test_unreadable_thumbnail_skips_the_container(self) -> None:
        with (
            patch(
                "django.db.models.fields.files.FieldFile.path",
                new_callable=PropertyMock,
                side_effect=ValueError("no path"),
            ),
            patch("api.classification.caption_generator.generate") as generate,
        ):
            self.assertFalse(generate_photo_caption(self.caption))
        generate.assert_not_called()

    def test_commit_false_neither_persists_caption_nor_writes_metadata(self) -> None:
        with (
            patch(
                "api.classification.caption_generator.generate",
                return_value=("A cat", "cat"),
            ),
            patch("api.metadata.photo_writer.write_metadata") as write,
            self.captureOnCommitCallbacks(execute=True),
        ):
            self.assertTrue(generate_photo_caption(self.caption, commit=False))
        self.assertEqual("A cat", self.caption.captions_json["im2txt"])
        self.assertIsNone(PhotoCaption.objects.get(pk=self.caption.pk).captions_json)
        write.assert_not_called()

    def test_an_edit_made_during_inference_is_preserved(self) -> None:
        def reply(path: str, ext: str) -> tuple[str, str]:
            edits.set_caption(self.photo, "My own caption", self.user)
            return "Generated caption", "cat"

        with patch("api.classification.caption_generator.generate", side_effect=reply):
            self.assertTrue(generate_photo_caption(self.caption))
        self.caption.refresh_from_db()
        self.assertEqual("My own caption", self.caption.captions_json["user_caption"])
        self.assertEqual("Generated caption", self.caption.captions_json["im2txt"])

    def test_automatic_generation_skips_complete_outputs(self) -> None:
        for legacy_pending in (None, False, True):
            with self.subTest(legacy_pending=legacy_pending):
                self.caption.captions_json = {"im2txt": "A cat", "im2txt_tag": ""}
                if legacy_pending is not None:
                    self.caption.captions_json["generated_metadata_pending"] = (
                        legacy_pending
                    )
                self.caption.save()
                with (
                    patch("api.classification.caption_generator.generate") as generate,
                    self.captureOnCommitCallbacks() as callbacks,
                ):
                    self.assertTrue(
                        generate_external_caption(self.caption, force=False)
                    )
                generate.assert_not_called()
                self.assertEqual([], callbacks)

    @override_settings(FEATURE_IMAGE_CAPTIONING=False)
    def test_tags_only_generation_preserves_an_existing_caption(self) -> None:
        self.caption.captions_json = {"im2txt": "Existing caption"}
        self.caption.save()
        with patch(
            "api.classification.caption_generator.generate",
            return_value=("New caption", "cat"),
        ):
            self.assertTrue(generate_external_caption(self.caption))
        self.caption.refresh_from_db()
        self.assertEqual("Existing caption", self.caption.captions_json["im2txt"])
        self.assertEqual("cat", self.caption.captions_json["im2txt_tag"])
