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

    def open_images(self, img_paths):
        paths = img_paths if isinstance(img_paths, list) else [img_paths]
        imgs = []
        for path in paths:
            image = None
            try:
                image = PIL.Image.open(path)
                image.load()
                imgs.append(image)
            except PIL.UnidentifiedImageError:
                if image is not None:
                    image.close()
                print(f"Error loading image: {path}")
            except Exception as e:
                for opened_image in (*imgs, image):
                    if opened_image is None:
                        continue
                    try:
                        opened_image.close()
                    except Exception:
                        pass
                print(f"Error loading image {path}: {e}")
                raise
        return imgs

    def batch_embeddings(self, imgs_emb):
        magnitudes = np.linalg.norm(imgs_emb, axis=1)
        return [row for row in imgs_emb], magnitudes.tolist()

    def single_embedding(self, imgs_emb):
        img_emb = imgs_emb[0].tolist()
        magnitude = float(np.linalg.norm(img_emb))
        return img_emb, magnitude

    def calculate_clip_embeddings(self, img_paths, model):
        if not self.model_is_loaded:
            self.load(model)
        imgs = self.open_images(img_paths)

        try:
            with torch.inference_mode():
                imgs_emb_tensor = self.model.encode(
                    imgs, batch_size=16, convert_to_tensor=True
                )
                imgs_emb_np = imgs_emb_tensor.detach().cpu().numpy()
            del imgs_emb_tensor

            if isinstance(img_paths, list):
                result = self.batch_embeddings(imgs_emb_np)
            else:
                result = self.single_embedding(imgs_emb_np)

            del imgs_emb_np
            return result
        except Exception as e:
            print(f"Error in calculating clip embeddings: {e}")
            raise
        finally:
            for image in imgs:
                try:
                    image.close()
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
