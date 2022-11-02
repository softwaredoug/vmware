try:
    import path  # noqa: F401
except ImportError:
    pass

import pandas as pd
from vmware.vector.corpus import scan_to_dataframe, passage_dataframe, quantize
from elasticsearch import Elasticsearch
from vmware.vector.mpnet import encode as encode_mpnet


def quantize_encode_mpnet(text):
    arr = encode_mpnet(text)
    return quantize(arr)


def build_mpnet_corpus(es):
    try:
        corpus = pd.read_pickle('corpus.pkl')
    except IOError:
        corpus = scan_to_dataframe(es, fields=['first_line', 'raw_text'])
        corpus.to_pickle('corpus.pkl')
    corpus = passage_dataframe(corpus, 'raw_text', quantize_encode_mpnet)
    corpus.to_pickle('corpus_mpnet.pkl')


if __name__ == "__main__":
    es = Elasticsearch()
    build_mpnet_corpus(es)
