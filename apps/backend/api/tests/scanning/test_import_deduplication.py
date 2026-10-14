"""Scan groups, uploads and concurrent imports reuse the same Photo."""

import io
import queue
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch

from django import db
from django.db import connection, transaction
from django.test import TestCase, TransactionTestCase, override_settings
from PIL import Image

from api.directory_watcher.file_handlers import create_new_image, group_files_into_photo
from api.directory_watcher.scan_jobs import _partition_scan_paths
from api.models import File, FileCopy, Photo
from api.tests.utils import create_test_file, create_test_photo, create_test_user


@override_settings(FEATURE_PROCESS_EMBEDDED_MEDIA=False)
class ImportDeduplicationTest(TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.user = create_test_user()
        content = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(content, format="JPEG")
        self.content = content.getvalue()
        self.file = create_test_file(
            str(self.root / "original.jpg"), self.user, content=self.content
        )

    def test_single_file_imports_of_identical_content_reuse_the_photo(self) -> None:
        copy = self.root / "copy.jpg"
        copy.write_bytes(self.content)
        first = create_new_image(self.user, self.file.path)
        second = create_new_image(self.user, str(copy))
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(1, Photo.objects.owned_by(self.user).count())
        self.assertEqual(1, File.objects.count())
        self.file.refresh_from_db()
        self.assertEqual(str(self.root / "original.jpg"), self.file.path)
        self.assertTrue(
            FileCopy.objects.filter(path=str(copy), file=self.file).exists()
        )

    def test_image_hash_reclaims_a_removed_photo_without_file_links(self) -> None:
        photo = create_test_photo(
            owner=self.user, main_file=self.file, removed=True, in_trashcan=True
        )
        Photo.objects.filter(pk=photo.pk).update(main_file=None)
        result = group_files_into_photo(self.user, [self.file], "rescan")
        self.assertEqual(photo.pk, result.pk)
        result.refresh_from_db()
        self.assertFalse(result.removed)
        self.assertFalse(result.in_trashcan)
        self.assertIsNone(result.trashed_at)
        self.assertEqual(self.file.pk, result.main_file_id)
        self.assertEqual([self.file], list(result.files.all()))

    def test_a_photo_without_a_main_file_adopts_the_main_file(self) -> None:
        photo = create_test_photo(owner=self.user, main_file=self.file)
        photo.files.add(self.file)
        Photo.objects.filter(pk=photo.pk).update(main_file=None, video=True)
        result = group_files_into_photo(self.user, [self.file], "rescan")
        self.assertEqual(photo.pk, result.pk)
        self.assertEqual(self.file.pk, result.main_file_id)
        self.assertFalse(result.video)

    def test_replacing_a_video_main_file_with_an_image_clears_the_video_flag(
        self,
    ) -> None:
        video = create_test_file(
            str(self.root / "original.mov"), self.user, file_type=File.VIDEO
        )
        photo = create_test_photo(owner=self.user, main_file=video, video=True)
        photo.files.add(video)
        result = group_files_into_photo(self.user, [video, self.file], "rescan")
        self.assertEqual(photo.pk, result.pk)
        self.assertEqual(self.file.pk, result.main_file_id)
        self.assertFalse(result.video)

    def test_repeated_paths_are_collected_only_once(self) -> None:
        image = self.file.path
        sidecar = str(self.root / "original.xmp")
        groups, metadata = _partition_scan_paths([image, sidecar, image, sidecar])
        self.assertEqual([[image, sidecar]], list(groups.values()))
        self.assertEqual([], metadata)

    def test_a_new_user_starts_exactly_one_scan_after_commit(self) -> None:
        (self.root / "alice").mkdir()
        with (
            override_settings(USER_HOME_DIRECTORY_PREFIX=str(self.root)),
            patch(
                "api.accounts.home_scan.missing_models_prerequisites", return_value=[]
            ),
            patch("api.jobs.queue.threading.Thread") as thread,
            self.captureOnCommitCallbacks(execute=True),
        ):
            user = create_test_user(username="alice")
            thread.assert_not_called()
        self.assertEqual(str(self.root / "alice"), user.scan_directory)
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()


@skipUnless(
    connection.vendor == "postgresql", "concurrent row locking requires PostgreSQL"
)
@override_settings(FEATURE_PROCESS_EMBEDDED_MEDIA=False)
class ConcurrentPhotoImportTest(TransactionTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.user = create_test_user()
        content = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(content, format="JPEG")
        self.file = create_test_file(
            str(Path(temporary.name) / "photo.jpg"),
            self.user,
            content=content.getvalue(),
        )

    def _import_twice(self, single_file: bool) -> None:
        pids: queue.Queue[int] = queue.Queue()

        def import_photo() -> object:
            try:
                with db.connection.cursor() as cursor:
                    cursor.execute("SELECT pg_backend_pid()")
                    pids.put(cursor.fetchone()[0])
                photo = (
                    create_new_image(self.user, self.file.path)
                    if single_file
                    else group_files_into_photo(self.user, [self.file], "concurrent")
                )
                return photo.pk
            finally:
                db.connections.close_all()

        with ThreadPoolExecutor(max_workers=2) as pool:
            # Both imports must reach a lock before this transaction releases
            # the file. Without import serialization, both see no Photo and
            # block on the INSERT's foreign key check instead, then create two.
            with transaction.atomic():
                File.objects.select_for_update().get(pk=self.file.pk)
                futures = [pool.submit(import_photo) for _ in range(2)]
                worker_pids = [pids.get(timeout=10) for _ in range(2)]
                deadline = time.monotonic() + 10
                blocked = 0
                while time.monotonic() < deadline:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT COUNT(DISTINCT pid) FROM pg_locks WHERE pid = ANY(%s) AND NOT granted",
                            [worker_pids],
                        )
                        blocked = cursor.fetchone()[0]
                    if blocked == 2:
                        break
                    time.sleep(0.01)
                self.assertEqual(
                    2, blocked, "both imports must be waiting for the locked file"
                )
            photo_ids = [future.result(timeout=10) for future in futures]
        self.assertEqual(photo_ids[0], photo_ids[1])
        self.assertEqual(1, Photo.objects.owned_by(self.user).count())
        self.assertEqual(1, File.objects.count())

    def test_concurrent_groups_create_one_photo(self) -> None:
        self._import_twice(single_file=False)

    def test_concurrent_uploads_create_one_photo(self) -> None:
        self._import_twice(single_file=True)
