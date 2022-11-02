from sentence_transformers import SentenceTransformer
# import requests_cache
import requests
from urllib.parse import urljoin
import numpy as np
import re

model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)


_mpnet_mapping = {
  "properties": {
    "raw_text_mean_mpnet": {
        "type": "dense_vector",
        "dims": 768
    },
  }
}

s = requests.Session()


def _mpnet_passages_server_encode(lines, passages_url='http://192.168.1.47:5001'):
    encoded_lines = []
    for line in lines:
        model_url = urljoin(passages_url, f"encode/{model_name}")
        resp = s.get(model_url, params={'q': line})
        if resp.status_code > 300:
            raise RuntimeError(f"Bad request - {resp.status} - {passages_url}")
        encoded_lines.append(resp.json()['encoded'])
    mean = np.mean(np.array(encoded_lines), axis=0)
    assert mean.shape == (768,)
    return mean


def _mpnet_encode_lines(lines):
    encoded = model.encode(lines)
    return np.mean(encoded)


def _mpnet_encode(doc_source, encoder=_mpnet_passages_server_encode):
    """First lines appear to be titles?"""
    lines = [doc_source['first_line']]
    any_alpha = re.compile('[a-zA-Z]')
    for line in doc_source['remaining_lines']:
        if len(line) > 10 and re.search(any_alpha, line):
            lines.append(line)
            if len(lines) > 100:
                break

    doc_source['raw_text_mean_mpnet'] = encoder(lines).tolist()

    return doc_source


mapping = _mpnet_mapping
enrichment = _mpnet_encode
