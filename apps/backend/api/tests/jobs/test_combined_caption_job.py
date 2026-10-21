"""Scan follow-up selects missing outputs, without metadata persistence retries."""

import uuid
from unittest.mock import patch

from constance.test import override_config
from django.test import TestCase, override_settings

from api.directory_watcher.processing_jobs import (
    generate_im2txt_captions,
    generate_im2txt_job,
)
from api.models import LongRunningJob
from api.tests.utils import create_test_photo, create_test_user


class CombinedCaptionJobTest(TestCase):
    def setUp(self):
        self.user = create_test_user()

    @override_config(CAPTIONING_MODEL="None")
    def test_only_missing_outputs_are_selected_regardless_of_legacy_write_markers(self):
        caption_missing = create_test_photo(
            owner=self.user,
            captions_json={"im2txt_tag": "cat"},
        )
        tags_missing = create_test_photo(
            owner=self.user,
            captions_json={"im2txt": "A cat"},
        )
        create_test_photo(
            owner=self.user,
            captions_json={
                "im2txt": "A cat",
                "im2txt_tag": "cat",
                "generated_metadata_pending": True,
            },
        )
        create_test_photo(
            owner=self.user,
            captions_json={
                "im2txt": "Done",
                "im2txt_tag": "",
            },
        )
        create_test_photo(owner=create_test_user())
        with patch("api.directory_watcher.processing_jobs.enqueue_task") as enqueue:
            generate_im2txt_captions(self.user, uuid.uuid4())
        self.assertEqual(
            {caption_missing.pk, tags_missing.pk},
            {call.args[1] for call in enqueue.call_args_list},
        )

    def test_a_worker_calls_the_external_generator_for_both_outputs_once(self):
        photo = create_test_photo(owner=self.user)
        job_id = uuid.uuid4()
        with patch("api.directory_watcher.processing_jobs.enqueue_task"):
            generate_im2txt_captions(self.user, job_id)
        with patch(
            "api.classification.caption_generator.generate",
            return_value=("A cat", "cat, sofa"),
        ) as generate:
            generate_im2txt_job(photo.pk, job_id)
        generate.assert_called_once()
        photo.refresh_from_db()
        self.assertEqual("A cat", photo.caption_instance.captions_json["im2txt"])
        self.assertEqual(
            "cat, sofa", photo.caption_instance.captions_json["im2txt_tag"]
        )
        job = LongRunningJob.objects.get(job_id=job_id)
        self.assertTrue(job.finished)
        self.assertEqual(1, job.progress_current)

    @override_settings(FEATURE_IMAGE_CAPTIONING=False)
    def test_scene_classification_alone_still_queues_the_combined_worker(self):
        photo = create_test_photo(owner=self.user)
        with patch("api.directory_watcher.processing_jobs.enqueue_task") as enqueue:
            generate_im2txt_captions(self.user, uuid.uuid4())
        enqueue.assert_called_once()
        self.assertEqual(photo.pk, enqueue.call_args.args[1])

    @override_settings(
        FEATURE_IMAGE_CAPTIONING=False, FEATURE_SCENE_CLASSIFICATION=False
    )
    def test_both_features_off_completes_without_queuing_inference(self):
        create_test_photo(owner=self.user)
        job_id = uuid.uuid4()
        with patch("api.directory_watcher.processing_jobs.enqueue_task") as enqueue:
            generate_im2txt_captions(self.user, job_id)
        enqueue.assert_not_called()
        self.assertTrue(LongRunningJob.objects.get(job_id=job_id).finished)
