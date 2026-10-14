"""Missing scans delete physical ghosts and their thumbnails in bounded batches."""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from django.db import transaction
from django.test import TestCase, override_settings

from api.directory_watcher.scan_jobs import _bulk_delete_photos, scan_missing_photos
from api.models import LongRunningJob, Photo, Thumbnail
from api.models.thumbnail import thumbnail_file_paths
from api.tests.utils import (
    create_test_album_thing,
    create_test_file,
    create_test_photo,
    create_test_tag,
    create_test_user,
)


class BulkDeleteMissingPhotosTest(TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings = override_settings(MEDIA_ROOT=str(self.root / "thumbnails"))
        settings.enable()
        self.addCleanup(settings.disable)
        self.user = create_test_user(scan_directory=str(self.root))
        self.missing_file = create_test_file(str(self.root / "missing.jpg"), self.user)
        self.photo = create_test_photo(owner=self.user, main_file=self.missing_file)
        self.photo.files.add(self.missing_file)
        self.paths = [
            Path(path) for path in thumbnail_file_paths(self.photo.image_hash)
        ]
        self.paths.append(self.paths[0].with_suffix(".mp4"))
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"thumbnail")

    def test_scan_deletes_photo_and_all_thumbnails_even_when_a_sidecar_survives(
        self,
    ) -> None:
        sidecar = create_test_file(
            str(self.root / "missing.xmp"), self.user, content=b"<xmp/>"
        )
        self.photo.files.add(sidecar)
        job_id = uuid.uuid4()
        with self.captureOnCommitCallbacks(execute=True):
            scan_missing_photos(self.user, job_id)
            self.assertTrue(all(path.exists() for path in self.paths))
        self.assertFalse(Photo.objects.filter(pk=self.photo.pk).exists())
        self.assertFalse(Thumbnail.objects.filter(pk=self.photo.pk).exists())
        self.assertTrue(Path(sidecar.path).exists())
        self.assertFalse(any(path.exists() for path in self.paths))
        job = LongRunningJob.objects.get(job_id=job_id)
        self.assertTrue(job.finished)
        self.assertFalse(job.failed)
        self.assertEqual({"deleted_photos": 1}, job.result)

    def test_thumbnail_files_are_removed_without_a_thumbnail_row(self) -> None:
        Thumbnail.objects.filter(photo=self.photo).delete()
        with self.captureOnCommitCallbacks(execute=True):
            self.assertEqual(1, _bulk_delete_photos([self.photo.pk], self.user))
        self.assertFalse(any(path.exists() for path in self.paths))

    def test_other_users_and_reappeared_media_are_not_deleted(self) -> None:
        other = create_test_photo(
            owner=create_test_user(), main_file=create_test_file(None)
        )
        Path(self.missing_file.path).write_bytes(b"the file returned")
        with self.captureOnCommitCallbacks(execute=True):
            self.assertEqual(
                0, _bulk_delete_photos([self.photo.pk, other.pk], self.user)
            )
        self.assertTrue(Photo.objects.filter(pk=self.photo.pk).exists())
        self.assertTrue(Photo.objects.filter(pk=other.pk).exists())
        self.assertTrue(all(path.exists() for path in self.paths))

    def test_a_surviving_media_variant_keeps_the_photo(self) -> None:
        variant = create_test_file(
            str(self.root / "photo.CR2"), self.user, content=b"raw"
        )
        self.photo.files.add(variant)
        with self.captureOnCommitCallbacks(execute=True):
            self.assertEqual(0, _bulk_delete_photos([self.photo.pk], self.user))
        self.assertTrue(Photo.objects.filter(pk=self.photo.pk).exists())

    def test_shared_thumbnail_files_survive(self) -> None:
        other = create_test_photo(
            owner=create_test_user(), main_file=create_test_file(None)
        )
        Photo.objects.filter(pk=other.pk).update(image_hash=self.photo.image_hash)
        with self.captureOnCommitCallbacks(execute=True):
            self.assertEqual(1, _bulk_delete_photos([self.photo.pk], self.user))
        self.assertTrue(all(path.exists() for path in self.paths))

    def test_a_thumbnail_reference_keeps_legacy_previews_without_a_matching_hash(
        self,
    ) -> None:
        other = create_test_photo(owner=create_test_user())
        legacy = self.paths[-1]
        Thumbnail.objects.filter(photo=other).update(
            thumbnail_big=str(legacy.relative_to(self.root / "thumbnails"))
        )
        with self.captureOnCommitCallbacks(execute=True):
            self.assertEqual(1, _bulk_delete_photos([self.photo.pk], self.user))
        self.assertTrue(legacy.exists())

    def test_a_rolled_back_bulk_delete_keeps_rows_and_files(self) -> None:
        with self.captureOnCommitCallbacks(execute=True):
            with self.assertRaises(RuntimeError), transaction.atomic():
                _bulk_delete_photos([self.photo.pk], self.user)
                raise RuntimeError("roll back the deletion")
        self.assertTrue(Photo.objects.filter(pk=self.photo.pk).exists())
        self.assertTrue(all(path.exists() for path in self.paths))

    def test_bulk_deletion_refreshes_thing_and_tag_counts(self) -> None:
        kept = create_test_photo(owner=self.user)
        thing = create_test_album_thing(owner=self.user)
        thing.photos.add(self.photo, kept)
        tag = create_test_tag(self.user)
        tag.photos.add(self.photo, kept)
        _bulk_delete_photos([self.photo.pk], self.user)
        thing.refresh_from_db()
        tag.refresh_from_db()
        self.assertEqual(1, thing.photo_count)
        self.assertEqual(1, tag.photo_count)

    def test_scan_deletes_in_batches_and_reports_the_total(self) -> None:
        for index in range(2):
            file = create_test_file(str(self.root / f"missing-{index}.jpg"), self.user)
            create_test_photo(owner=self.user, main_file=file).files.add(file)
        job_id = uuid.uuid4()
        with (
            patch("api.directory_watcher.scan_jobs._BULK_DELETE_BATCH_SIZE", 2),
            patch(
                "api.directory_watcher.scan_jobs._bulk_delete_photos",
                wraps=_bulk_delete_photos,
            ) as delete,
        ):
            scan_missing_photos(self.user, job_id)
        self.assertEqual([2, 1], [len(call.args[0]) for call in delete.call_args_list])
        self.assertFalse(Photo.objects.owned_by(self.user).exists())
        self.assertEqual(
            3, LongRunningJob.objects.get(job_id=job_id).result["deleted_photos"]
        )
