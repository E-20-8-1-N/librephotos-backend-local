"""The fork's home-path API and scan rules on the current upstream backend."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from constance.test import override_config
from django.test import TestCase, override_settings
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response
from rest_framework.test import APIClient

from api.accounts.home_scan import start_home_scan, user_from_path
from api.directory_watcher.utils import walk_directory, walk_files
from api.models import LongRunningJob, User
from api.models.photo_search import place_label
from api.tests.utils import create_test_job, create_test_user


class HomeScanTest(TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.client = APIClient()

    def test_blank_directory_is_assigned_but_explicit_directory_is_kept(self) -> None:
        with override_settings(USER_HOME_DIRECTORY_PREFIX=self.root):
            user = create_test_user(username="alice")
            explicit = create_test_user(username="bob", scan_directory="/other/photos")
        self.assertEqual(os.path.join(self.root, "alice"), user.scan_directory)
        self.assertEqual("/other/photos", explicit.scan_directory)
        self.assertEqual(User.SaveMetadata.MEDIA_FILE, user.save_metadata_to_disk)

    def test_new_home_scan_waits_for_the_user_transaction_to_commit(self) -> None:
        Path(self.root, "alice").mkdir()
        with (
            override_settings(USER_HOME_DIRECTORY_PREFIX=self.root),
            patch("api.accounts.home_scan.start_home_scan") as start,
            self.captureOnCommitCallbacks(execute=True),
        ):
            user = create_test_user(username="alice")
            start.assert_not_called()
        start.assert_called_once()
        self.assertEqual(user.pk, start.call_args.args[0].pk)

    def test_path_resolution_rejects_sibling_prefix_and_symlink_escape(self) -> None:
        with override_settings(USER_HOME_DIRECTORY_PREFIX=self.root):
            create_test_user(username="alice")
            Path(self.root, "alice").mkdir()
            Path(self.root, "alice", "outside").symlink_to(os.path.dirname(self.root))
            for path in (
                self.root + "-sibling/alice/photo.jpg",
                os.path.join(self.root, "alice", "outside", "file.jpg"),
            ):
                with self.subTest(path=path), self.assertRaises(PermissionDenied):
                    user_from_path(path)

    def test_anonymous_backend_scan_keeps_path_api_and_immediate_job_response(
        self,
    ) -> None:
        with override_settings(USER_HOME_DIRECTORY_PREFIX=self.root):
            user = create_test_user(username="alice")
            Path(user.scan_directory).mkdir()
            photo_path = Path(user.scan_directory, "photo.jpg")
            photo_path.touch()
            job = create_test_job(started_by=user)
            with patch(
                "api.views.scan_triggers.start_home_scan", return_value=job
            ) as start:
                response = self.client.post(
                    "/api/backendscanphotos", {"path": str(photo_path)}, format="json"
                )
        self.assertEqual(200, response.status_code)
        self.assertTrue(response.data["status"])
        self.assertEqual("alice", response.data["username"])
        start.assert_called_once()

    def test_uploaded_photo_scan_is_incremental(self) -> None:
        Path(self.root, "alice").mkdir()
        user = create_test_user(scan_directory=os.path.join(self.root, "alice"))
        self.client.force_authenticate(user)
        with patch("api.views.scan_triggers._start_photo_scan") as start:
            start.return_value = Response({})
            self.client.post("/api/scanuploadedphotos/", {}, format="json")
        self.assertEqual((user, user.scan_directory), start.call_args.args)
        self.assertEqual({}, start.call_args.kwargs)

    def test_queue_status_is_anonymous_without_exposing_another_users_job(self) -> None:
        user = create_test_user()
        create_test_job(started_by=user, job_type=LongRunningJob.JOB_SCAN_PHOTOS)
        response = self.client.get("/api/rqavailable/")
        self.assertEqual(200, response.status_code)
        self.assertFalse(response.data["queue_can_accept_job"])
        self.assertIsNone(response.data["job_detail"])

    def test_immediate_scan_has_a_committed_job_and_honors_cancellation(self) -> None:
        user = create_test_user(scan_directory=self.root)
        with (
            patch(
                "api.accounts.home_scan.missing_models_prerequisites", return_value=[]
            ),
            patch("api.accounts.home_scan.scan_photos", autospec=True) as scan,
            patch("api.jobs.queue.threading.Thread") as thread,
            self.captureOnCommitCallbacks(execute=True),
        ):
            job = start_home_scan(user)
            self.assertTrue(LongRunningJob.objects.filter(pk=job.pk).exists())
            thread.assert_not_called()
        target = thread.call_args.kwargs["target"]
        target()
        self.assertTrue(scan.call_args.kwargs["inline_image_processing"])
        self.assertTrue(scan.call_args.kwargs["force_im2txt"])
        scan.reset_mock()
        LongRunningJob.objects.filter(pk=job.pk).update(cancelled=True)
        target()
        scan.assert_not_called()


class ScanExclusionsTest(TestCase):
    @override_config(SCAN_SKIP_EXTENSIONS="pdf, .MKV")
    def test_walk_and_explicit_paths_share_exclusions_and_allowed_hidden_folder(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root:
            shared = Path(root, ".cetapod")
            shared.mkdir()
            visible = shared / "photo.jpg"
            ignored = shared / "clip.MKV"
            document = shared / "document.PDF"
            for path in (visible, ignored, document):
                path.touch()
            walked: list[str] = []
            explicit: list[str] = []
            walk_directory(root, walked)
            walk_files([str(visible), str(ignored), str(document)], explicit)
        self.assertEqual([str(visible)], walked)
        self.assertEqual(walked, explicit)

    def test_location_label_appends_places_once_and_keeps_fallback(self) -> None:
        self.assertEqual(
            "Sydney, Bondi",
            place_label({"address": "Sydney", "places": ["Sydney", "Bondi", "Bondi"]}),
        )
        self.assertEqual("Sydney", place_label({"features": [{"text": "Sydney"}]}))
