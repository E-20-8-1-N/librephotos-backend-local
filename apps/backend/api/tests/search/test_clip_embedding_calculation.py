"""Tests for ``SemanticSearch.calculate_clip_embeddings``.

The encoder and Pillow opener are mocked, while real CPU tensors exercise the
Torch-to-NumPy conversion used for both CPU and CUDA model output.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import PIL.Image
import torch
from django.test import SimpleTestCase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from service.clip_embeddings.semantic_search.semantic_search import (  # noqa: E402
    SemanticSearch,
)

MODULE = "service.clip_embeddings.semantic_search.semantic_search"


def make_embeddings(rows):
    return torch.tensor(rows, dtype=torch.float32)


class CalculateClipEmbeddingsTestCase(SimpleTestCase):
    def setUp(self):
        self.search = SemanticSearch()
        self.model = MagicMock()
        self.model.encode.return_value = make_embeddings([[3.0, 4.0], [6.0, 8.0]])
        self.search.model = self.model
        self.search.model_is_loaded = True

    def test_model_is_loaded_lazily_when_flag_is_false(self):
        search = SemanticSearch()
        search.model_is_loaded = False
        fake_model = MagicMock()
        fake_model.encode.return_value = make_embeddings([[3.0, 4.0]])
        image = MagicMock(name="image")

        with patch(f"{MODULE}.SentenceTransformer", return_value=fake_model) as loader:
            with patch("PIL.Image.open", return_value=image):
                search.calculate_clip_embeddings("/photo.jpg", "clip-model-name")

        loader.assert_called_once_with("clip-model-name")
        self.assertIs(search.model, fake_model)
        self.assertTrue(search.model_is_loaded)
        image.load.assert_called_once_with()
        image.close.assert_called_once_with()

    def test_model_is_not_reloaded_when_already_loaded(self):
        image = MagicMock(name="image")
        with patch(f"{MODULE}.SentenceTransformer") as loader:
            with patch("PIL.Image.open", return_value=image):
                self.search.calculate_clip_embeddings("/photo.jpg", "unused-model")

        loader.assert_not_called()
        self.assertIs(self.search.model, self.model)

    def test_list_input_on_cpu_returns_numpy_rows_and_reusable_lists(self):
        image_a = MagicMock(name="image_a")
        image_b = MagicMock(name="image_b")
        with patch("PIL.Image.open", side_effect=[image_a, image_b]):
            with patch("torch.cuda.is_available", return_value=False):
                with patch("torch.cuda.empty_cache") as empty_cache:
                    embeddings, magnitudes = self.search.calculate_clip_embeddings(
                        ["/a.jpg", "/b.jpg"], "m"
                    )

        self.model.encode.assert_called_once_with(
            [image_a, image_b], batch_size=16, convert_to_tensor=True
        )
        self.assertIsInstance(embeddings, list)
        self.assertTrue(all(type(row) is np.ndarray for row in embeddings))
        self.assertEqual([row.tolist() for row in embeddings], [[3.0, 4.0], [6.0, 8.0]])
        self.assertIsInstance(magnitudes, list)
        self.assertTrue(all(type(value) is float for value in magnitudes))
        self.assertEqual(magnitudes, [5.0, 10.0])
        self.assertEqual(
            [row.tolist() for row in embeddings],
            [
                [3.0, 4.0],
                [6.0, 8.0],
            ],
        )
        self.assertEqual(list(magnitudes), [5.0, 10.0])
        self.assertEqual(list(magnitudes), [5.0, 10.0])
        image_a.load.assert_called_once_with()
        image_b.load.assert_called_once_with()
        image_a.close.assert_called_once_with()
        image_b.close.assert_called_once_with()
        empty_cache.assert_not_called()

    def test_single_path_on_cpu_returns_plain_lists_and_float(self):
        image = MagicMock(name="image")
        with patch("PIL.Image.open", return_value=image) as opener:
            with patch("torch.cuda.is_available", return_value=False):
                embedding, magnitude = self.search.calculate_clip_embeddings(
                    "/only.jpg", "m"
                )

        opener.assert_called_once_with("/only.jpg")
        self.model.encode.assert_called_once_with(
            [image], batch_size=16, convert_to_tensor=True
        )
        self.assertIsInstance(embedding, list)
        self.assertEqual(embedding, [3.0, 4.0])
        self.assertIs(type(magnitude), float)
        self.assertEqual(magnitude, 5.0)

    def test_list_subclass_is_treated_as_a_batch(self):
        class PathList(list):
            pass

        paths = PathList(["/a.jpg", "/b.jpg"])
        image_a = MagicMock(name="image_a")
        image_b = MagicMock(name="image_b")
        with patch("PIL.Image.open", side_effect=[image_a, image_b]) as opener:
            with patch("torch.cuda.is_available", return_value=False):
                embeddings, magnitudes = self.search.calculate_clip_embeddings(
                    paths, "m"
                )

        self.assertEqual(opener.call_args_list, [call("/a.jpg"), call("/b.jpg")])
        self.model.encode.assert_called_once_with(
            [image_a, image_b], batch_size=16, convert_to_tensor=True
        )
        self.assertIsInstance(embeddings, list)
        self.assertTrue(all(isinstance(row, np.ndarray) for row in embeddings))
        self.assertEqual(magnitudes, [5.0, 10.0])

    def test_tuple_input_remains_a_single_pillow_path(self):
        paths = ("/a.jpg", "/b.jpg")
        image = MagicMock(name="image")
        with patch("PIL.Image.open", return_value=image) as opener:
            with patch("torch.cuda.is_available", return_value=False):
                embedding, magnitude = self.search.calculate_clip_embeddings(paths, "m")

        opener.assert_called_once_with(paths)
        self.model.encode.assert_called_once_with(
            [image], batch_size=16, convert_to_tensor=True
        )
        self.assertEqual(embedding, [3.0, 4.0])
        self.assertEqual(magnitude, 5.0)

    def test_pathlib_path_is_treated_as_a_single_path(self):
        path = Path("/tmp/pic.jpg")
        image = MagicMock(name="image")
        with patch("PIL.Image.open", return_value=image) as opener:
            with patch("torch.cuda.is_available", return_value=False):
                embedding, _ = self.search.calculate_clip_embeddings(path, "m")

        opener.assert_called_once_with(path)
        self.assertEqual(embedding, [3.0, 4.0])

    def test_cuda_batch_has_the_same_public_types_as_cpu(self):
        image_a = MagicMock(name="image_a")
        image_b = MagicMock(name="image_b")
        with patch("PIL.Image.open", side_effect=[image_a, image_b]):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.empty_cache") as empty_cache:
                    embeddings, magnitudes = self.search.calculate_clip_embeddings(
                        ["/a.jpg", "/b.jpg"], "m"
                    )

        self.assertIsInstance(embeddings, list)
        self.assertTrue(all(type(row) is np.ndarray for row in embeddings))
        self.assertEqual([row.tolist() for row in embeddings], [[3.0, 4.0], [6.0, 8.0]])
        self.assertIsInstance(magnitudes, list)
        self.assertTrue(all(type(value) is float for value in magnitudes))
        self.assertEqual(magnitudes, [5.0, 10.0])
        empty_cache.assert_called_once_with()

    def test_cuda_single_has_the_same_public_types_as_cpu(self):
        image = MagicMock(name="image")
        with patch("PIL.Image.open", return_value=image):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.empty_cache") as empty_cache:
                    embedding, magnitude = self.search.calculate_clip_embeddings(
                        "/only.jpg", "m"
                    )

        self.assertIsInstance(embedding, list)
        self.assertEqual(embedding, [3.0, 4.0])
        self.assertIs(type(magnitude), float)
        self.assertEqual(magnitude, 5.0)
        empty_cache.assert_called_once_with()

    def test_unidentified_image_in_a_batch_is_skipped_and_printed(self):
        good = MagicMock(name="good")
        self.model.encode.return_value = make_embeddings([[3.0, 4.0]])
        with patch(
            "PIL.Image.open",
            side_effect=[good, PIL.UnidentifiedImageError("nope")],
        ):
            with patch("builtins.print") as printer:
                with patch("torch.cuda.is_available", return_value=False):
                    embeddings, magnitudes = self.search.calculate_clip_embeddings(
                        ["/ok.jpg", "/bad.jpg"], "m"
                    )

        self.model.encode.assert_called_once_with(
            [good], batch_size=16, convert_to_tensor=True
        )
        self.assertEqual([row.tolist() for row in embeddings], [[3.0, 4.0]])
        self.assertEqual(magnitudes, [5.0])
        printer.assert_called_once_with("Error loading image: /bad.jpg")
        good.close.assert_called_once_with()

    def test_unidentified_single_image_encodes_empty_list_then_reraises(self):
        self.model.encode.return_value = make_embeddings([]).reshape(0, 2)
        with patch("PIL.Image.open", side_effect=PIL.UnidentifiedImageError("nope")):
            with patch("builtins.print") as printer:
                with patch("torch.cuda.is_available", return_value=False):
                    with self.assertRaises(IndexError):
                        self.search.calculate_clip_embeddings("/bad.jpg", "m")

        self.model.encode.assert_called_once_with(
            [], batch_size=16, convert_to_tensor=True
        )
        self.assertEqual(
            printer.call_args_list,
            [
                call("Error loading image: /bad.jpg"),
                call(
                    "Error in calculating clip embeddings: index 0 is out of bounds for axis 0 with size 0"
                ),
            ],
        )

    def test_empty_list_returns_two_empty_reusable_lists(self):
        self.model.encode.return_value = make_embeddings([]).reshape(0, 2)
        with patch("PIL.Image.open") as opener:
            with patch("torch.cuda.is_available", return_value=False):
                embeddings, magnitudes = self.search.calculate_clip_embeddings([], "m")

        opener.assert_not_called()
        self.model.encode.assert_called_once_with(
            [], batch_size=16, convert_to_tensor=True
        )
        self.assertEqual(embeddings, [])
        self.assertEqual(magnitudes, [])
        self.assertEqual(list(embeddings), [])
        self.assertEqual(list(magnitudes), [])

    def test_non_image_format_open_error_is_reraised(self):
        boom = FileNotFoundError("missing")
        with patch("PIL.Image.open", side_effect=boom):
            with patch("builtins.print") as printer:
                with self.assertRaises(FileNotFoundError) as context:
                    self.search.calculate_clip_embeddings(["/gone.jpg"], "m")

        self.assertIs(context.exception, boom)
        self.model.encode.assert_not_called()
        printer.assert_called_once_with("Error loading image /gone.jpg: missing")

    def test_open_error_closes_current_and_previously_loaded_images(self):
        good = MagicMock(name="good")
        broken = MagicMock(name="broken")
        boom = OSError("decode failed")
        broken.load.side_effect = boom

        with patch("PIL.Image.open", side_effect=[good, broken]):
            with patch("builtins.print") as printer:
                with self.assertRaises(OSError) as context:
                    self.search.calculate_clip_embeddings(
                        ["/good.jpg", "/broken.jpg"], "m"
                    )

        self.assertIs(context.exception, boom)
        good.load.assert_called_once_with()
        broken.load.assert_called_once_with()
        good.close.assert_called_once_with()
        broken.close.assert_called_once_with()
        self.model.encode.assert_not_called()
        printer.assert_called_once_with(
            "Error loading image /broken.jpg: decode failed"
        )

    def test_encode_failure_is_printed_reraised_and_fully_cleaned_up(self):
        image = MagicMock(name="image")
        boom = RuntimeError("CUDA OOM")
        self.model.encode.side_effect = boom

        with patch("PIL.Image.open", return_value=image):
            with patch("builtins.print") as printer:
                with patch(f"{MODULE}.gc.collect") as collect:
                    with patch("torch.cuda.is_available", return_value=True):
                        with patch("torch.cuda.empty_cache") as empty_cache:
                            with self.assertRaises(RuntimeError) as context:
                                self.search.calculate_clip_embeddings(["/a.jpg"], "m")

        self.assertIs(context.exception, boom)
        self.model.encode.assert_called_once_with(
            [image], batch_size=16, convert_to_tensor=True
        )
        image.close.assert_called_once_with()
        collect.assert_called_once_with()
        empty_cache.assert_called_once_with()
        printer.assert_called_once_with(
            "Error in calculating clip embeddings: CUDA OOM"
        )

    def test_close_failure_does_not_skip_remaining_cleanup(self):
        image_a = MagicMock(name="image_a")
        image_b = MagicMock(name="image_b")
        image_a.close.side_effect = OSError("close failed")

        with patch("PIL.Image.open", side_effect=[image_a, image_b]):
            with patch(f"{MODULE}.gc.collect") as collect:
                with patch("torch.cuda.is_available", return_value=False):
                    self.search.calculate_clip_embeddings(["/a.jpg", "/b.jpg"], "m")

        image_a.close.assert_called_once_with()
        image_b.close.assert_called_once_with()
        collect.assert_called_once_with()
