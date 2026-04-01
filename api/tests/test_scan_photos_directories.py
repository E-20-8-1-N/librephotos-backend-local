import os
import tempfile
import uuid
from unittest.mock import patch

from constance.test import override_config
from django.test import TestCase, override_settings

from api.directory_watcher import scan_photos
from api.directory_watcher.utils import walk_directory, walk_files
from api.tests.utils import create_test_user


class DummyAsyncTask:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        return None


class DummyChain:
    def __init__(self, *args, **kwargs):
        self.appended = []

    def append(self, *args, **kwargs):
        self.appended.append((args, kwargs))
        return self

    def run(self):
        return None


class ScanPhotosDirectoryCreationTest(TestCase):
    def test_existing_thumbnail_directory_does_not_raise(self):
        user = create_test_user()
        with tempfile.TemporaryDirectory() as media_root:
            preexisting_dir = os.path.join(media_root, "square_thumbnails_small")
            os.makedirs(preexisting_dir, exist_ok=True)

            user.scan_directory = media_root
            user.save(update_fields=["scan_directory"])

            with override_settings(MEDIA_ROOT=media_root):
                with (
                    patch("api.directory_watcher.scan_jobs.walk_directory"),
                    patch("api.directory_watcher.scan_jobs.walk_files"),
                    patch("api.directory_watcher.scan_jobs.photo_scanner"),
                    patch("api.directory_watcher.scan_jobs.AsyncTask", DummyAsyncTask),
                    patch("api.directory_watcher.scan_jobs.Chain", DummyChain),
                    patch("api.directory_watcher.scan_jobs.db.connections.close_all"),
                ):
                    scan_photos(user, full_scan=False, job_id=str(uuid.uuid4()))

            expected_directories = [
                "square_thumbnails_small",
                "square_thumbnails",
                "thumbnails_big",
            ]
            for directory_name in expected_directories:
                directory_path = os.path.join(media_root, directory_name)
                self.assertTrue(
                    os.path.isdir(directory_path),
                    msg=f"Expected directory {directory_path} to exist",
                )


class ScanPhotosDirectoryWalkTest(TestCase):
    def test_walk_directory_includes_cetapod_share(self):
        with tempfile.TemporaryDirectory() as scan_root:
            allowed_hidden_dir = os.path.join(scan_root, ".cetapod_share")
            skipped_hidden_dir = os.path.join(scan_root, ".hidden")
            visible_dir = os.path.join(scan_root, "visible")

            os.makedirs(allowed_hidden_dir, exist_ok=True)
            os.makedirs(skipped_hidden_dir, exist_ok=True)
            os.makedirs(visible_dir, exist_ok=True)

            allowed_file = os.path.join(allowed_hidden_dir, "shared.jpg")
            skipped_file = os.path.join(skipped_hidden_dir, "hidden.jpg")
            visible_file = os.path.join(visible_dir, "visible.jpg")

            for file_path in (allowed_file, skipped_file, visible_file):
                with open(file_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write("test")

            collected_paths = []
            walk_directory(scan_root, collected_paths)

            self.assertIn(allowed_file, collected_paths)
            self.assertIn(visible_file, collected_paths)
            self.assertNotIn(skipped_file, collected_paths)

    @override_config(SCAN_SKIP_EXTENSIONS=".pdf,.mkv,.avi,.wmv,.flv")
    def test_walk_directory_skips_configured_extensions(self):
        with tempfile.TemporaryDirectory() as scan_root:
            visible_dir = os.path.join(scan_root, "visible")
            os.makedirs(visible_dir, exist_ok=True)

            kept_file = os.path.join(visible_dir, "kept.jpg")
            skipped_files = [
                os.path.join(visible_dir, "skip.pdf"),
                os.path.join(visible_dir, "skip.mkv"),
                os.path.join(visible_dir, "skip.AVI"),
                os.path.join(visible_dir, "skip.wmv"),
                os.path.join(visible_dir, "skip.FLV"),
            ]

            for file_path in [kept_file, *skipped_files]:
                with open(file_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write("test")

            collected_paths = []
            walk_directory(scan_root, collected_paths)

            self.assertIn(kept_file, collected_paths)
            for skipped_file in skipped_files:
                self.assertNotIn(skipped_file, collected_paths)

    @override_config(SCAN_SKIP_EXTENSIONS=".pdf,.mkv,.avi,.wmv,.flv")
    def test_walk_files_skips_configured_extensions(self):
        with tempfile.TemporaryDirectory() as scan_root:
            kept_file = os.path.join(scan_root, "kept.jpg")
            skipped_files = [
                os.path.join(scan_root, "skip.pdf"),
                os.path.join(scan_root, "skip.mkv"),
                os.path.join(scan_root, "skip.avi"),
                os.path.join(scan_root, "skip.WMV"),
                os.path.join(scan_root, "skip.flv"),
            ]

            for file_path in [kept_file, *skipped_files]:
                with open(file_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write("test")

            collected_paths = []
            walk_files([kept_file, *skipped_files], collected_paths)

            self.assertEqual(collected_paths, [kept_file])

    @override_config(SCAN_SKIP_EXTENSIONS="pdf,avi")
    def test_walk_files_uses_runtime_configured_extensions(self):
        with tempfile.TemporaryDirectory() as scan_root:
            kept_files = [
                os.path.join(scan_root, "keep.jpg"),
                os.path.join(scan_root, "keep.mkv"),
            ]
            skipped_files = [
                os.path.join(scan_root, "skip.pdf"),
                os.path.join(scan_root, "skip.AVI"),
            ]

            for file_path in [*kept_files, *skipped_files]:
                with open(file_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write("test")

            collected_paths = []
            walk_files([*kept_files, *skipped_files], collected_paths)

            self.assertEqual(collected_paths, kept_files)
