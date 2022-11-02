import elasticsearch.helpers
import pandas as pd
import numpy as np
import multiprocessing as mp


def quantize(arr, bits=256):
    """Scale to 0-255, cast to uint8."""
    floor = -1.0
    ceil = 1.0
    arr[arr == 1.0] = ceil - 0.00001
    assert not (arr > ceil).any()
    assert not (arr < floor).any()
    flt_per_bucket = (abs(floor) + abs(ceil)) / bits
    quantized = (arr - floor) // flt_per_bucket
    assert not (arr >= bits).any()
    return quantized.astype(np.uint8)


def scan_to_dataframe(es, index='vmware', fields=['first_line'], n=None):
    if 'id' not in fields:
        fields.append('id')
    search_scroll_body = {
        "query": {
            "match_all": {}
        },
        "_source": fields
    }
    docs = []
    for idx, doc in enumerate(elasticsearch.helpers.scan(es, index=index,
                                                         scroll='5m',
                                                         size=200,
                                                         request_timeout=120,
                                                         query=search_scroll_body)):
        if idx % 100 == 0:
            print(f"Scanned {idx}")
        docs.append(doc['_source'])
        if n is not None:
            if len(docs) > n:
                break
    return pd.DataFrame(docs)


def passage_dataframe(corpus, column, encoder):
    expl_column = column + "_passages"
    corpus[expl_column] = corpus[column].str.replace('\r', ' ')
    corpus[expl_column] = corpus[expl_column].str.split('\n')
    corpus = corpus.explode(expl_column)

    print(f"Encoding {len(corpus)} passages")

    with mp.Pool(mp.cpu_count()) as pool:

        corpus[column + "_passages_vector"] = pool.map(
            encoder,
            corpus[expl_column]
        )
    return corpus
