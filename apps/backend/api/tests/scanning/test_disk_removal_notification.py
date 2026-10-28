"""Filesystem deletion notifications clean records before returning to the caller."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TransactionTestCase, override_settings
from rest_framework.test import APIClient

from api.models import File, FileCopy, LongRunningJob, Photo, Thumbnail
from api.models.face import Face
from api.models.photo_caption import PhotoCaption
from api.models.photo_metadata import PhotoMetadata
from api.models.photo_search import PhotoSearch
from api.models.thumbnail import thumbnail_file_paths
from api.tests.utils import (
    create_test_face,
    create_test_file,
    create_test_job,
    create_test_photo,
    create_test_user,
)


class DiskRemovalNotificationTest(TransactionTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "alice"
        self.home.mkdir()
        self.user = create_test_user(username="alice", scan_directory=str(self.home))
        settings = override_settings(
            USER_HOME_DIRECTORY_PREFIX=str(self.root),
            MEDIA_ROOT=str(self.root / "media"),
        )
        settings.enable()
        self.addCleanup(settings.disable)
        self.client = APIClient()
        self.file = create_test_file(
            str(self.home / "photo.jpg"), self.user, on_disk=True
        )
        self.photo = create_test_photo(
            owner=self.user,
            main_file=self.file,
            captions_json={"im2txt": "A photo"},
            search_captions="A photo",
            iso=100,
        )
        self.photo.files.add(self.file)
        self.thumbnails = [
            Path(path) for path in thumbnail_file_paths(self.photo.image_hash)
        ]
        for path in self.thumbnails:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"thumbnail")

    def notify(self, path: Path):
        with patch("api.jobs.queue.AsyncTask") as queued:
            response = self.client.post(
                "/api/backenddeletemissingphotos/", {"path": str(path)}, format="json"
            )
        self.assertEqual(200, response.status_code, response.content)
        queued.assert_not_called()
        return response

    def test_removed_file_is_deleted_from_database_and_thumbnails_immediately(
        self,
    ) -> None:
        path = Path(self.file.path)
        path.unlink()
        response = self.notify(path)
        self.assertFalse(Photo.objects.filter(pk=self.photo.pk).exists())
        self.assertFalse(File.objects.filter(pk=self.file.pk).exists())
        for model in (Thumbnail, PhotoCaption, PhotoMetadata, PhotoSearch):
            self.assertFalse(model.objects.filter(photo_id=self.photo.pk).exists())
        self.assertFalse(any(path.exists() for path in self.thumbnails))
        job = LongRunningJob.objects.get(job_id=response.data["job_id"])
        self.assertTrue(job.finished)
        self.assertEqual(1, job.result["deleted_photos"])

    def test_removal_notification_is_idempotent(self) -> None:
        path = Path(self.file.path)
        path.unlink()
        self.notify(path)
        again = self.notify(path)
        self.assertEqual(
            0,
            LongRunningJob.objects.get(job_id=again.data["job_id"]).result[
                "deleted_photos"
            ],
        )

    def test_existing_delete_job_does_not_delay_a_removal_event(self) -> None:
        existing = create_test_job(
            started_by=self.user, job_type=LongRunningJob.JOB_DELETE_MISSING_PHOTOS
        )
        path = Path(self.file.path)
        path.unlink()
        self.notify(path)
        self.assertFalse(Photo.objects.filter(pk=self.photo.pk).exists())
        existing.refresh_from_db()
        self.assertFalse(existing.finished)

    def test_a_recreated_file_is_kept(self) -> None:
        self.notify(Path(self.file.path))
        self.assertTrue(Photo.objects.filter(pk=self.photo.pk).exists())
        self.assertTrue(File.objects.filter(pk=self.file.pk).exists())
        self.assertTrue(all(path.exists() for path in self.thumbnails))

    def test_a_removed_folder_is_scoped_without_matching_its_sibling(self) -> None:
        folder = self.home / "holiday"
        sibling = self.home / "holiday-old"
        folder.mkdir()
        sibling.mkdir()
        gone = create_test_file(str(folder / "one.jpg"), self.user, on_disk=True)
        gone_photo = create_test_photo(owner=self.user, main_file=gone)
        gone_photo.files.add(gone)
        other = create_test_file(str(sibling / "two.jpg"), self.user, on_disk=True)
        other_photo = create_test_photo(owner=self.user, main_file=other)
        other_photo.files.add(other)
        Path(gone.path).unlink()
        folder.rmdir()
        Path(other.path).unlink()
        self.notify(folder)
        self.assertFalse(Photo.objects.filter(pk=gone_photo.pk).exists())
        self.assertTrue(Photo.objects.filter(pk=other_photo.pk).exists())
        self.assertTrue(Photo.objects.filter(pk=self.photo.pk).exists())

    def test_a_surviving_media_variant_becomes_the_main_file(self) -> None:
        raw = create_test_file(
            str(self.home / "photo.CR2"), self.user, content=b"raw image"
        )
        self.photo.files.add(raw)
        path = Path(self.file.path)
        path.unlink()
        self.notify(path)
        self.photo.refresh_from_db()
        self.assertEqual(raw.pk, self.photo.main_file_id)
        self.assertEqual([raw.pk], list(self.photo.files.values_list("pk", flat=True)))
        self.assertFalse(File.objects.filter(pk=self.file.pk).exists())
        self.assertTrue(all(path.exists() for path in self.thumbnails))

    def test_an_exact_copy_keeps_the_photo_and_replaces_the_missing_path(self) -> None:
        copy = self.home / "copy.jpg"
        create_test_file(
            str(copy), self.user, content=Path(self.file.path).read_bytes()
        )
        path = Path(self.file.path)
        path.unlink()
        self.notify(path)
        self.photo.refresh_from_db()
        self.assertEqual(str(copy), self.photo.main_file.path)
        self.assertFalse(FileCopy.objects.filter(path=str(copy)).exists())

    def test_embedded_motion_preview_does_not_keep_a_removed_photo_alive(self) -> None:
        preview_path = self.root / "media" / "motion.mp4"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = create_test_file(
            str(preview_path), self.user, content=b"derived video"
        )
        self.file.embedded_media.add(preview)
        self.photo.files.add(preview)
        path = Path(self.file.path)
        path.unlink()
        self.notify(path)
        self.assertFalse(Photo.objects.filter(pk=self.photo.pk).exists())
        self.assertFalse(File.objects.filter(pk=preview.pk).exists())
        self.assertFalse(preview_path.exists())

    def test_face_crops_and_photo_relations_are_removed(self) -> None:
        face = create_test_face(photo=self.photo, image="faces/deleted.jpg")
        crop = Path(face.image.path)
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"face crop")
        path = Path(self.file.path)
        path.unlink()
        self.notify(path)
        self.assertFalse(Face.objects.filter(pk=face.pk).exists())
        self.assertFalse(crop.exists())

    def test_another_users_photo_is_never_cleaned_by_a_notification(self) -> None:
        other_home = self.root / "bob"
        other_user = create_test_user(username="bob", scan_directory=str(other_home))
        other_home.mkdir()
        other_file = create_test_file(str(other_home / "removed.jpg"), other_user)
        other_photo = create_test_photo(owner=other_user, main_file=other_file)
        other_photo.files.add(other_file)
        path = Path(self.file.path)
        path.unlink()
        self.notify(path)
        self.assertTrue(Photo.objects.filter(pk=other_photo.pk).exists())
        self.assertTrue(File.objects.filter(pk=other_file.pk).exists())

    def test_a_removed_copy_record_is_cleaned_without_deleting_the_original(
        self,
    ) -> None:
        copy = self.home / "copy.jpg"
        create_test_file(
            str(copy), self.user, content=Path(self.file.path).read_bytes()
        )
        copy.unlink()
        self.notify(copy)
        self.assertFalse(FileCopy.objects.filter(path=str(copy)).exists())
        self.assertTrue(Photo.objects.filter(pk=self.photo.pk).exists())
        self.assertTrue(Path(self.file.path).exists())

    def test_orphaned_file_records_are_removed_after_a_prior_photo_cleanup(
        self,
    ) -> None:
        path = Path(self.file.path)
        path.unlink()
        Photo.objects.filter(pk=self.photo.pk).delete()
        self.notify(path)
        self.assertFalse(File.objects.filter(pk=self.file.pk).exists())
