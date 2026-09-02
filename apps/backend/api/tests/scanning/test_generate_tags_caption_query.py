from django.test import TestCase

from api.directory_watcher.processing_jobs import _untagged_photos
from api.models.photo_caption import PhotoCaption
from api.tests.utils import create_test_photo, create_test_user


class DirectoryWatcherFixTest(TestCase):
    def setUp(self):
        self.user = create_test_user()

    def test_generate_tags_query_works(self):
        """Photos without caption-generator tags remain pending."""
        photo = create_test_photo(owner=self.user)

        caption_instance, created = PhotoCaption.objects.get_or_create(photo=photo)
        caption_instance.captions_json = {
            "im2txt": "A beautiful landscape",
            "user_caption": "My vacation photo",
        }
        caption_instance.save()

        existing_photos = _untagged_photos(self.user)

        self.assertEqual(existing_photos.count(), 1)
        self.assertEqual(existing_photos.first(), photo)

    def test_generate_tags_query_excludes_caption_generator_tags(self):
        """Photos with ``im2txt_tag`` data are already complete."""
        photo = create_test_photo(owner=self.user)

        caption_instance, created = PhotoCaption.objects.get_or_create(photo=photo)
        caption_instance.captions_json = {"im2txt_tag": "outdoor, sunny"}
        caption_instance.save()

        existing_photos = _untagged_photos(self.user)

        self.assertEqual(existing_photos.count(), 0)
