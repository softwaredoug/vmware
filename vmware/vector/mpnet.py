from sentence_transformers import SentenceTransformer
import requests
import numpy as np
import os
from time import perf_counter
from urllib.parse import urljoin

model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

s = requests.Session()


def _mpnet_passages_server_encode(text, passages_url='http://192.168.1.47:5001'):
    model_url = urljoin(passages_url, f"encode/{model_name}")
    resp = s.get(model_url, params={'q': text})
    if resp.status_code > 300:
        raise RuntimeError(f"Bad request - {resp.status} - {passages_url}")

    _mpnet_passages_server_encode.call_count += 1

    if _mpnet_passages_server_encode.call_count % 1000 == 0:
        print(f"Mpnet{os.getpid()} - {_mpnet_passages_server_encode.call_count} - {perf_counter() - _mpnet_passages_server_encode.start }")

    return np.array(resp.json()['encoded'])


encode = _mpnet_passages_server_encode
_mpnet_passages_server_encode.call_count = 0
_mpnet_passages_server_encode.start = perf_counter()
