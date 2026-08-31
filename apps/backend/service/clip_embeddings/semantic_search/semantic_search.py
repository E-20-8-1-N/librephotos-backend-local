import gc

import numpy as np
import PIL
import torch
from pillow_heif import register_heif_opener
from sentence_transformers import SentenceTransformer

register_heif_opener()


class SemanticSearch:
    model = None
    model_is_loaded = False

    def load(self, model):
        self.load_model(model)
        self.model_is_loaded = True
        pass

    def unload(self):
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_is_loaded = False
        pass

    def load_model(self, model):
        self.model = SentenceTransformer(model)

    def calculate_clip_embeddings(self, img_paths, model):
        if not self.model_is_loaded:
            self.load(model)
        imgs = []
        is_batch = isinstance(img_paths, list)
        paths = img_paths if is_batch else [img_paths]
        for path in paths:
            try:
                img = PIL.Image.open(path)
                img.load()  # Force pixel data into memory
                imgs.append(img)
            except PIL.UnidentifiedImageError:
                print(f"Error loading image: {path}")
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        try:
            with torch.inference_mode():
                imgs_emb_tensor = self.model.encode(
                    imgs, batch_size=16, convert_to_tensor=True
                )
                # Move to CPU numpy ASAP and release the GPU/torch tensor so
                # a 64-image batch does not stay resident between calls.
                imgs_emb_np = imgs_emb_tensor.detach().cpu().numpy()
            del imgs_emb_tensor

            magnitudes = np.linalg.norm(imgs_emb_np, axis=1)

            if is_batch:
                result = ([row for row in imgs_emb_np], magnitudes.tolist())
            else:
                result = (imgs_emb_np[0].tolist(), float(magnitudes[0]))

            del imgs_emb_np
            return result
        except Exception as e:
            print(f"Error in calculating clip embeddings: {e}")
            raise
        finally:
            for img in imgs:
                try:
                    img.close()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def calculate_query_embeddings(self, query, model):
        if not self.model_is_loaded:
            self.load(model)

        with torch.inference_mode():
            q_tensor = self.model.encode([query], convert_to_tensor=True)
            query_emb = q_tensor[0].detach().cpu().numpy().tolist()
        del q_tensor
        magnitude = np.linalg.norm(query_emb)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return query_emb, magnitude
