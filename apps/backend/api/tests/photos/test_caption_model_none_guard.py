"""A legacy None model setting does not gate the external caption container."""

from unittest.mock import patch

from constance.test import override_config
from django.test import TestCase

from api.classification.captioning import generate_photo_caption
from api.models import PhotoCaption
from api.tests.utils import create_test_photo, create_test_user


class CaptionModelNoneTest(TestCase):
    def test_both_none_spellings_generate_through_the_container(self):
        photo = create_test_photo(owner=create_test_user())
        caption, _ = PhotoCaption.objects.get_or_create(photo=photo)
        for model in ("none", "None"):
            with (
                self.subTest(model=model),
                override_config(CAPTIONING_MODEL=model),
                patch(
                    "api.classification.caption_generator.generate",
                    return_value=("A cat", "cat"),
                ) as generate,
            ):
                self.assertTrue(generate_photo_caption(caption, commit=False))
            generate.assert_called_once()
