"""Tests for PhotoCaption's external caption-generator integration."""

from unittest.mock import PropertyMock, patch

from django.test import TestCase, override_settings

from api.models import PhotoCaption
from api.models.album_thing import AlbumThing
from api.tests.utils import create_test_photo, create_test_user


@override_settings(FEATURE_IMAGE_CAPTIONING=True)
class ExternalCaptionGenerationTest(TestCase):
    def setUp(self):
        self.user = create_test_user()
        self.photo = create_test_photo(owner=self.user)
        self.caption = PhotoCaption.objects.create(photo=self.photo)

    @override_settings(FEATURE_IMAGE_CAPTIONING=False)
    def test_feature_flag_disabled_returns_false(self):
        with patch("api.models.photo_caption.generate_image_caption") as generate:
            result = self.caption.generate_captions_im2txt(commit=False)

        self.assertFalse(result)
        generate.assert_not_called()
        self.assertIsNone(self.caption.captions_json)

    def test_empty_thumbnail_returns_false(self):
        thumbnail = self.photo.thumbnail
        thumbnail.thumbnail_big = ""
        thumbnail.save(update_fields=["thumbnail_big"])

        with patch("api.models.photo_caption.generate_image_caption") as generate:
            result = self.caption.generate_captions_im2txt(commit=False)

        self.assertFalse(result)
        generate.assert_not_called()

    def test_unreadable_thumbnail_path_returns_false(self):
        with (
            patch(
                "django.db.models.fields.files.FieldFile.path",
                new_callable=PropertyMock,
                side_effect=ValueError("no path"),
            ),
            patch("api.models.photo_caption.generate_image_caption") as generate,
        ):
            result = self.caption.generate_captions_im2txt(commit=False)

        self.assertFalse(result)
        generate.assert_not_called()

    def test_caption_and_tags_are_saved_and_indexed_as_album_things(self):
        with (
            patch(
                "api.models.photo_caption.generate_image_caption",
                return_value=("a cat on a sofa", "cat, pet"),
            ) as generate,
            patch.object(PhotoCaption, "recreate_search_captions") as recreate,
        ):
            result = self.caption.generate_captions_im2txt(commit=True)

        self.assertTrue(result)
        generate.assert_called_once_with(
            self.photo.thumbnail.thumbnail_big.path, ".webp"
        )
        recreate.assert_called_once_with()
        self.caption.refresh_from_db()
        self.assertEqual(self.caption.captions_json["im2txt"], "a cat on a sofa")
        self.assertEqual(self.caption.captions_json["im2txt_tag"], "cat, pet")
        self.assertEqual(
            set(
                AlbumThing.objects.filter(
                    owner=self.user,
                    photos=self.photo,
                    thing_type="caption_generator_tag",
                ).values_list("title", flat=True)
            ),
            {"cat", "pet"},
        )

    def test_commit_false_only_mutates_the_instance(self):
        with (
            patch(
                "api.models.photo_caption.generate_image_caption",
                return_value=("not persisted", None),
            ),
            patch.object(PhotoCaption, "recreate_search_captions"),
        ):
            result = self.caption.generate_captions_im2txt(commit=False)

        self.assertTrue(result)
        self.assertEqual(self.caption.captions_json["im2txt"], "not persisted")
        self.assertIsNone(PhotoCaption.objects.get(pk=self.caption.pk).captions_json)

    def test_existing_caption_keys_are_preserved(self):
        self.caption.captions_json = {"user_caption": "mine", "im2txt": "old"}
        self.caption.save()

        with (
            patch(
                "api.models.photo_caption.generate_image_caption",
                return_value=("new caption", None),
            ),
            patch.object(PhotoCaption, "recreate_search_captions"),
        ):
            self.assertTrue(self.caption.generate_captions_im2txt(commit=True))

        self.caption.refresh_from_db()
        self.assertEqual(self.caption.captions_json["user_caption"], "mine")
        self.assertEqual(self.caption.captions_json["im2txt"], "new caption")

    @override_settings(FEATURE_SCENE_CLASSIFICATION=False)
    def test_caption_generation_ignores_tags_when_tagging_is_disabled(self):
        with (
            patch(
                "api.models.photo_caption.generate_image_caption",
                return_value=("a cat", "cat, pet"),
            ),
            patch.object(PhotoCaption, "recreate_search_captions"),
        ):
            result = self.caption.generate_captions_im2txt(commit=True)

        self.assertTrue(result)
        self.caption.refresh_from_db()
        self.assertEqual(self.caption.captions_json["im2txt"], "a cat")
        self.assertNotIn("im2txt_tag", self.caption.captions_json)
        self.assertFalse(
            AlbumThing.objects.filter(thing_type="caption_generator_tag").exists()
        )

    def test_empty_generator_result_returns_false_without_mutating_state(self):
        with (
            patch(
                "api.models.photo_caption.generate_image_caption",
                return_value=(None, None),
            ),
            patch.object(PhotoCaption, "recreate_search_captions") as recreate,
        ):
            result = self.caption.generate_captions_im2txt(commit=True)

        self.assertFalse(result)
        self.assertIsNone(self.caption.captions_json)
        recreate.assert_not_called()

    def test_generator_exception_returns_false(self):
        with patch(
            "api.models.photo_caption.generate_image_caption",
            side_effect=RuntimeError("generator unavailable"),
        ):
            result = self.caption.generate_captions_im2txt(commit=True)

        self.assertFalse(result)
        self.assertIsNone(self.caption.captions_json)


@override_settings(FEATURE_SCENE_CLASSIFICATION=True)
class ExternalTagGenerationTest(TestCase):
    def setUp(self):
        self.user = create_test_user()
        self.photo = create_test_photo(owner=self.user)
        self.caption = PhotoCaption.objects.create(photo=self.photo)

    def test_tag_generation_uses_im2txt_tag_key(self):
        with (
            patch(
                "api.models.photo_caption.generate_image_caption",
                return_value=("a beach", "beach, sea"),
            ),
            patch.object(PhotoCaption, "recreate_search_captions"),
        ):
            result = self.caption.generate_tag_captions(commit=True)

        self.assertTrue(result)
        self.caption.refresh_from_db()
        self.assertEqual(self.caption.captions_json["im2txt_tag"], "beach, sea")
        self.assertNotIn("tag", self.caption.captions_json)

    def test_empty_generator_result_leaves_state_unchanged(self):
        with patch(
            "api.models.photo_caption.generate_image_caption",
            return_value=(None, None),
        ):
            result = self.caption.generate_tag_captions(commit=True)

        self.assertFalse(result)
        self.assertIsNone(self.caption.captions_json)

    @override_settings(FEATURE_IMAGE_CAPTIONING=False)
    def test_tag_generation_ignores_caption_when_captioning_is_disabled(self):
        with (
            patch(
                "api.models.photo_caption.generate_image_caption",
                return_value=("a beach", "beach, sea"),
            ),
            patch.object(PhotoCaption, "recreate_search_captions"),
        ):
            result = self.caption.generate_tag_captions(commit=True)

        self.assertTrue(result)
        self.caption.refresh_from_db()
        self.assertEqual(self.caption.captions_json["im2txt_tag"], "beach, sea")
        self.assertNotIn("im2txt", self.caption.captions_json)
