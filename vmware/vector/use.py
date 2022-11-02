import tensorflow_text  # NOQA: F401
import tensorflow_hub as hub

_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")


def encode(text):
    return _use(text).numpy()[0]
