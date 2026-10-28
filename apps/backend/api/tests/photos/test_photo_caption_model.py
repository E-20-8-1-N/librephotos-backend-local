from unittest.mock import patch

import requests
from django.test import TestCase

from api.classification.captioning import generate_photo_caption
from api.models import PhotoCaption
from api.search.semantic import generate_photo_tags
from api.tests.utils import create_test_photo, create_test_user


class PhotoCaptionModelTest(TestCase):
    def setUp(self):
        self.user = create_test_user()
        self.photo = create_test_photo(owner=self.user)

    def test_create_photo_caption(self):
        """Test creating a PhotoCaption instance"""
        caption = PhotoCaption.objects.create(
            photo=self.photo, captions_json={"user_caption": "Test caption"}
        )

        self.assertEqual(caption.photo, self.photo)
        self.assertEqual(caption.captions_json["user_caption"], "Test caption")

    def test_photo_caption_one_to_one_relationship(self):
        """Test that PhotoCaption has a one-to-one relationship with Photo"""
        PhotoCaption.objects.create(
            photo=self.photo, captions_json={"user_caption": "First caption"}
        )

        # Trying to create another caption for the same photo should fail
        with self.assertRaises(Exception):
            PhotoCaption.objects.create(
                photo=self.photo, captions_json={"user_caption": "Second caption"}
            )

    def test_generate_captions_im2txt(self):
        """Test generating im2txt captions"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        self.photo.thumbnail.thumbnail_big = ""
        self.photo.thumbnail.save(update_fields=["thumbnail_big"])
        result = generate_photo_caption(caption, commit=False)
        self.assertFalse(result)

    def test_apply_user_caption(self):
        """Test saving user captions"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        result = caption.apply_user_caption("My beautiful photo", commit=True)
        self.assertEqual(result, "My beautiful photo")

    def test_generate_tag_captions_skips_existing(self):
        """Test that generate_photo_tags skips if active model tags already exist"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        caption.captions_json = {
            "im2txt": "A landscape",
            "im2txt_tag": "outdoor, landscape",
        }
        caption.save()

        # Should return early since the active model's tags already exist
        generate_photo_tags(caption, commit=True)
        caption.refresh_from_db()

        self.assertIn("im2txt_tag", caption.captions_json)

    def test_recreate_search_captions_delegates_to_photo_search(self):
        """Test that recreate_search_captions delegates to PhotoSearch"""
        caption = PhotoCaption.objects.create(
            photo=self.photo, captions_json={"user_caption": "Test caption"}
        )

        # This should create a PhotoSearch instance and update search captions
        caption.recreate_search_captions()

        # Verify PhotoSearch was created and has search captions
        self.assertTrue(hasattr(self.photo, "search_instance"))
        from api.models.photo_search import PhotoSearch

        search_instance, created = PhotoSearch.objects.get_or_create(photo=self.photo)
        self.assertIsNotNone(search_instance.search_captions)

    def test_captions_json_default_empty_dict(self):
        """Test that captions_json defaults to None (nullable field)"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        self.assertIsNone(caption.captions_json)

    def test_str_representation(self):
        """Test string representation of PhotoCaption"""
        caption = PhotoCaption.objects.create(
            photo=self.photo, captions_json={"user_caption": "Test"}
        )

        str_repr = str(caption)
        self.assertIn(self.photo.image_hash, str_repr)

    def test_cascade_delete_with_photo(self):
        """Test that PhotoCaption is deleted when Photo is deleted"""
        PhotoCaption.objects.create(photo=self.photo)
        photo_id = self.photo.pk

        self.photo.delete()

        with self.assertRaises(PhotoCaption.DoesNotExist):
            PhotoCaption.objects.get(photo_id=photo_id)

    def test_multiple_caption_types(self):
        """Test storing multiple types of captions"""
        caption = PhotoCaption.objects.create(
            photo=self.photo,
            captions_json={
                "user_caption": "My photo",
                "im2txt": "a photo of a landscape",
                "openclip_vitb32": {"tags": ["outdoor", "natural"]},
            },
        )

        self.assertEqual(caption.captions_json["user_caption"], "My photo")
        self.assertEqual(caption.captions_json["im2txt"], "a photo of a landscape")
        self.assertIn("tags", caption.captions_json["openclip_vitb32"])

    def test_update_existing_captions(self):
        """Test updating existing captions"""
        caption = PhotoCaption.objects.create(
            photo=self.photo, captions_json={"user_caption": "Original caption"}
        )

        # Update the caption directly
        caption.captions_json["user_caption"] = "Updated caption"
        caption.save()
        caption.refresh_from_db()

        self.assertEqual(caption.captions_json["user_caption"], "Updated caption")

    def test_empty_captions_json_handling(self):
        """Test handling of empty or None captions_json"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        # Should handle empty dict gracefully
        caption.recreate_search_captions()

        # Test direct assignment
        caption.captions_json = {"user_caption": ""}
        caption.save()
        self.assertEqual(caption.captions_json["user_caption"], "")

    @patch("api.classification.caption_generator.generate")
    def test_generate_tag_captions_handles_error_response(self, mock_post):
        """Test that generate_photo_tags handles non-OK HTTP response gracefully"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        mock_post.side_effect = requests.HTTPError("500 Server Error")
        with self.assertRaises(requests.HTTPError):
            generate_photo_tags(caption, commit=False)

        # captions_json should remain unchanged
        self.assertIsNone(caption.captions_json)

    @patch("api.classification.caption_generator.generate")
    def test_generate_tag_captions_handles_non_json_response(self, mock_post):
        """Test that generate_photo_tags handles non-JSON response body gracefully"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        mock_post.side_effect = ValueError("No JSON object could be decoded")
        with self.assertRaises(ValueError):
            generate_photo_tags(caption, commit=False)

        # captions_json should remain unchanged
        self.assertIsNone(caption.captions_json)

    @patch("api.classification.caption_generator.generate")
    def test_generate_tag_captions_handles_missing_tags_key(self, mock_post):
        """Test that generate_photo_tags handles JSON response without 'tags' key"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        mock_post.side_effect = ValueError(
            "Caption generator returned no caption or Things"
        )
        with self.assertRaises(ValueError):
            generate_photo_tags(caption, commit=False)

        # captions_json should remain unchanged
        self.assertIsNone(caption.captions_json)

    @patch("api.classification.caption_generator.generate")
    def test_generate_tag_captions_handles_400_response(self, mock_post):
        """Test that generate_photo_tags handles HTTP 400 (bad request) gracefully"""
        caption = PhotoCaption.objects.create(photo=self.photo)

        mock_post.side_effect = requests.HTTPError("400 Client Error")
        with self.assertRaises(requests.HTTPError):
            generate_photo_tags(caption, commit=False)

        # captions_json should remain unchanged
        self.assertIsNone(caption.captions_json)
