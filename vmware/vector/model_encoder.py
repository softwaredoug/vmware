from sentence_transformers import SentenceTransformer
import numpy as np

from .vector_cache import VectorCache


import redis
r = redis.Redis(host='localhost', port=6379)


class ModelEncoder:
    """A naive cache for SentenceTransformer inference.

    This should just use Redis :)
    """

    def __init__(self, model_name, dims):

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        model_base_name = model_name.replace("/", "_")
        self.cache = VectorCache(r, namespace=model_base_name,
                                 dtype=np.float32,
                                 dims=dims)

    def encode(self, text, cached=True):
        """Embed a single snippet, prefer cached version if available."""
        if cached:
            cached_value = self.cache.get(text)
            if cached_value is not None:
                return cached_value

        encoded = self.model.encode(text)
        self.cache.set(text, encoded)
        return encoded
