"""The external container's HTTP contract and failure behavior."""

from unittest.mock import Mock, patch

import requests
from django.test import SimpleTestCase, override_settings

from api.classification import caption_generator
from api.drf.exceptions import ServiceUnavailable


@override_settings(CAPTION_GENERATOR_RETRIES=0)
class GenerateCaptionTest(SimpleTestCase):
    def test_caption_and_tag_fields_are_returned_from_the_external_endpoint(
        self,
    ) -> None:
        response = Mock(spec=requests.Response)
        response.json.return_value = {"caption": "A dog", "tags": ["dog", "park"]}
        with (
            patch.object(caption_generator, "ensure_ready", return_value=True),
            patch.object(
                caption_generator.requests, "post", return_value=response
            ) as post,
        ):
            self.assertEqual(
                ("A dog", "dog, park"),
                caption_generator.generate("/photo.webp", ".webp"),
            )
        self.assertIn("caption-generator:8020/rushgenerate", post.call_args.args[0])
        self.assertEqual(
            {"file_path": "/photo.webp", "file_ext": ".webp"},
            post.call_args.kwargs["json"],
        )

    def test_connection_error_propagates_without_an_unbounded_retry(self) -> None:
        with (
            patch.object(caption_generator, "ensure_ready", return_value=True),
            patch.object(
                caption_generator.requests, "post", side_effect=requests.ConnectionError
            ),
            self.assertRaises(requests.ConnectionError),
        ):
            caption_generator.generate("/photo.webp", ".webp")

    def test_unhealthy_container_does_not_receive_generation_requests(self) -> None:
        with (
            patch.object(caption_generator, "ensure_ready", return_value=False),
            patch.object(caption_generator.requests, "post") as post,
            self.assertRaisesRegex(ServiceUnavailable, "unavailable"),
        ):
            caption_generator.generate("/photo.webp", ".webp")
        post.assert_not_called()
