import json
from .passage_similarity import passage_similarity_long_lines
from .splainer import splainer_url
from .query_cache import MemoizeQuery


def max_passage_rerank_first_remaining_lines(es, query):
    """Try only the earliest remaining lines for matching (NDCG, 0.29)."""
    body = {
        'size': 5,
        'query': {
            'bool': {'should': [
                {'match_phrase': {
                    'first_remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'first_remaining_lines': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] \
            = passage_similarity_long_lines(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


@MemoizeQuery
def rerank_simple_slop_search_max_snippet_at_5(es, query):
    """Rerank top 5 submissions by max passage USE similarity.
       NDCG of 0.30562."""
    body = {
        'size': 5,
        'query': {
            'bool': {'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity_long_lines(query, hit, verbose=False)
        hit['_source']['splainer'] = splainer_url(es_body=body)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


@MemoizeQuery
def rerank_simple_slop_search_sum_snippets_at_5(es, query):
    """Rerank top 5 submissions by SUM of passage USE similarity.
       NDCG of 0.30550"""
    body = {
        'size': 5,
        'query': {
            'bool': {'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity_long_lines(query, hit, verbose=False)
        hit['_source']['splainer'] = splainer_url(es_body=body)

    hits = sorted(hits, key=lambda x: x['_source']['sum_sim'], reverse=True)
    hits = hits[:5]
    return hits


@MemoizeQuery
def rerank_slop_search_max_passage_rerank_at_10(es, query):
    """Rerank top 50 submissions by max passage USE similarity"""
    body = {
        'size': 10,
        'query': {
            'bool': {'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] =\
            passage_similarity_long_lines(query, hit, verbose=False)
        hit['_source']['splainer'] = splainer_url(es_body=body)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


@MemoizeQuery
def max_passage_rerank_at_50(es, query):
    """Rerank top 50 submissions by max passage USE similarity"""
    body = {
        'size': 50,
        'query': {
            'bool': {'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] =\
            passage_similarity_long_lines(query, hit, verbose=False)
        hit['_source']['splainer'] = splainer_url(es_body=body)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits
