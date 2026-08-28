import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

from django.db import connections
from django.test import (
    TestCase,
    TransactionTestCase,
    override_settings,
    skipUnlessDBFeature,
)
from django.utils import timezone
from PIL import Image

from api.directory_watcher.file_handlers import (
    create_new_image,
    group_files_into_photo,
)
from api.models import File, Photo, User


class PhotoDeduplicationMixin:
    def setUp(self):
        super().setUp()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.user = User.objects.create(
            username="deduplication-test-user",
            save_metadata_to_disk=User.SaveMetadata.OFF,
        )

    def create_image(self, name):
        path = Path(self.temporary_directory.name, name)
        Image.new("RGB", (2, 2), color=(12, 34, 56)).save(path, format="PNG")
        return str(path)

    def create_photo(self, image_hash, main_file=None):
        return Photo.objects.create(
            image_hash=image_hash,
            owner=self.user,
            added_on=timezone.now(),
            geolocation_json={},
            main_file=main_file,
        )


@override_settings(FEATURE_PROCESS_EMBEDDED_MEDIA=False)
class PhotoDeduplicationTests(PhotoDeduplicationMixin, TestCase):
    def test_repeated_path_import_reuses_photo(self):
        path = self.create_image("same-path.png")

        first_photo = create_new_image(self.user, path)
        second_photo = create_new_image(self.user, path)

        self.assertEqual(first_photo.pk, second_photo.pk)
        self.assertEqual(Photo.objects.count(), 1)
        self.assertEqual(File.objects.count(), 1)

    def test_copied_file_reuses_hash_without_replacing_original_path(self):
        original_path = self.create_image("original.png")
        copied_path = str(Path(self.temporary_directory.name, "copy.png"))
        shutil.copyfile(original_path, copied_path)

        original_photo = create_new_image(self.user, original_path)
        copied_photo = create_new_image(self.user, copied_path)

        self.assertEqual(original_photo.pk, copied_photo.pk)
        self.assertEqual(Photo.objects.count(), 1)
        self.assertEqual(File.objects.count(), 1)
        self.assertEqual(File.objects.get().path, original_path)
        self.assertFalse(File.objects.filter(path=copied_path).exists())

    def test_image_hash_match_reuses_photo_without_attached_file(self):
        path = self.create_image("hash-match.png")
        file = File.create(path, self.user)
        existing_photo = self.create_photo(file.hash)

        photo = group_files_into_photo(self.user, [file], "test-job")

        self.assertEqual(photo.pk, existing_photo.pk)
        self.assertEqual(Photo.objects.count(), 1)
        self.assertTrue(photo.files.filter(pk=file.pk).exists())
        self.assertEqual(photo.main_file_id, file.pk)

    def test_secondary_file_hash_match_reuses_photo(self):
        image_path = self.create_image("variant-image.png")
        image_file = File.create(image_path, self.user)
        raw_file = File.objects.create(
            hash="raw-file-hash",
            path=str(Path(self.temporary_directory.name, "variant-image.raw")),
            type=File.RAW_FILE,
        )
        existing_photo = self.create_photo(raw_file.hash)

        photo = group_files_into_photo(self.user, [image_file, raw_file], "test-job")

        self.assertEqual(photo.pk, existing_photo.pk)
        self.assertEqual(Photo.objects.count(), 1)
        self.assertEqual(photo.files.count(), 2)
        self.assertEqual(photo.main_file_id, image_file.pk)

    def test_main_file_path_match_reuses_photo_without_file_relation(self):
        path = self.create_image("path-match.png")
        file = File.create(path, self.user)
        existing_photo = self.create_photo("legacy-image-hash", main_file=file)

        photo = group_files_into_photo(self.user, [file], "test-job")

        self.assertEqual(photo.pk, existing_photo.pk)
        self.assertEqual(Photo.objects.count(), 1)
        self.assertTrue(photo.files.filter(pk=file.pk).exists())

    def test_active_photo_is_preferred_over_older_removed_duplicate(self):
        path = self.create_image("active-match.png")
        file = File.create(path, self.user)
        removed_photo = self.create_photo(file.hash, main_file=file)
        removed_photo.added_on = timezone.now() - timedelta(days=1)
        removed_photo.removed = True
        removed_photo.in_trashcan = True
        removed_photo.save()
        active_photo = self.create_photo(file.hash, main_file=file)
        active_photo.files.add(file)

        photo = group_files_into_photo(self.user, [file], "test-job")

        removed_photo.refresh_from_db()
        self.assertEqual(photo.pk, active_photo.pk)
        self.assertTrue(removed_photo.removed)
        self.assertTrue(removed_photo.in_trashcan)

    def test_main_file_change_updates_video_flag(self):
        image_path = self.create_image("preferred-image.png")
        image_file = File.create(image_path, self.user)
        video_file = File.objects.create(
            hash="video-file-hash",
            path=str(Path(self.temporary_directory.name, "video.mov")),
            type=File.VIDEO,
        )
        existing_photo = self.create_photo("legacy-video-hash", main_file=video_file)
        existing_photo.video = True
        existing_photo.save()
        existing_photo.files.add(image_file, video_file)

        photo = group_files_into_photo(self.user, [image_file], "test-job")

        photo.refresh_from_db()
        self.assertEqual(photo.pk, existing_photo.pk)
        self.assertEqual(photo.main_file_id, image_file.pk)
        self.assertFalse(photo.video)


@override_settings(FEATURE_PROCESS_EMBEDDED_MEDIA=False)
class ConcurrentPhotoDeduplicationTests(PhotoDeduplicationMixin, TransactionTestCase):
    @skipUnlessDBFeature("has_select_for_update")
    def test_concurrent_hash_imports_create_one_photo(self):
        original_path = self.create_image("concurrent-original.png")
        copied_path = str(Path(self.temporary_directory.name, "concurrent-copy.png"))
        shutil.copyfile(original_path, copied_path)
        barrier = Barrier(2)

        def import_file(path):
            connections.close_all()
            try:
                user = User.objects.get(pk=self.user.pk)
                barrier.wait()
                return create_new_image(user, path).pk
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(import_file, path)
                for path in (original_path, copied_path)
            ]
            photo_ids = [future.result() for future in futures]

        self.assertEqual(photo_ids[0], photo_ids[1])
        self.assertEqual(Photo.objects.count(), 1)
        self.assertEqual(File.objects.count(), 1)
