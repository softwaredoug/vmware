from .passage_similarity import passage_similarity_long_lines
from .splainer import splainer_url
from .query_cache import MemoizeQuery

from vmware.search.compounds import most_freq_compound_strategy, to_compound_query, \
    most_freq_compound_corpus_strategy, most_freq_compound_both_strategy


def with_best_compounds_at_50_plus_10_times_use(es, query, params=None, rerank=True):
    """Shot in the dark on multiplying the USE score differently.

    adding 10* to BM25 instead of replacing BM25 score. Best on 5 - June NDCG 0.31643
    """
    if params is None:
        params = {
            'rerank_depth': 50,
            'remaining_lines_slop': 10,
            'first_line_slop': 10,
            'remaining_lines_phrase_boost': 1.0,
            'remaining_lines_phrase_decomp_boost': 1.0,
            'first_line_phrase_boost': 1.0,
            'first_line_phrase_decomp_boost': 1.0,
            'first_line_boost': 1.0,
            'remaining_lines_boost': 1.0,
            'use_weight': 10.0,
            'elasticsearch_score_weight': 1.0,
        }

    to_decompound, to_compound = most_freq_compound_strategy[0], most_freq_compound_strategy[1]
    body = {
        'size': 50,
        'query': {
            'bool': {'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': int(params['remaining_lines_slop']),
                        'boost': params['remaining_lines_phrase_boost'],
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': int(params['first_line_slop']),
                        'boost': params['first_line_boost'],
                        'query': query
                    }
                }},
                {'match': {
                    'remaining_lines': {
                        'query': query,
                        'boost': params['remaining_lines_boost'],
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query,
                        'boost': params['first_line_boost'],
                    }
                }},
            ]}
        }
    }

    new_query = to_compound_query(query, to_decompound, to_compound)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = \
                [{'match_phrase': {   # noqa: E127
                    'remaining_lines': {
                        'slop': 10,
                        'query': new_query,
                        'boost': params['remaining_lines_phrase_decomp_boost'],
                    }
                }},
                {'match_phrase': {   # noqa: E122
                    'first_line': {
                        'slop': 10,
                        'query': new_query,
                        'boost': params['first_line_phrase_decomp_boost'],
                    }
                }}]
        body['query']['bool']['should'].extend(alt_clauses)

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        passage_similarity_long_lines(query, hit, verbose=False)

    if rerank:
        hits = sorted(hits, key=lambda x:
                      ((params['use_weight'] * x['_source']['max_sim_use'])
                       + (params['elasticsearch_score_weight'] * x['_score'])), reverse=True)
        hits = hits[:5]
    return hits


with_best_compounds_at_50_plus_10_times_use.params = [
    'rerank_depth',
    'remaining_lines_slop',
    'first_line_slop',
    'remaining_lines_phrase_boost',
    'remaining_lines_phrase_decomp_boost',
    'first_line_phrase_boost',
    'first_line_phrase_decomp_boost',
    'first_line_boost',
    'remaining_lines_boost',
    'use_weight',
    'elasticsearch_score_weight'
]


# @MemoizeQuery
def with_best_compounds_at_5_only_phrase_search(es, query, params={}):
    """Add using compounds computed from query dataset.

    Second best submission on 5-June, improves a few qureies, NDCG - 0.31643
    """
    to_decompound, to_compound = most_freq_compound_strategy[0], most_freq_compound_strategy[1]
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
                    'remaining_lines': {
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

    new_query = to_compound_query(query, to_decompound, to_compound)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = \
                [{'match_phrase': {   # noqa: E127
                    'remaining_lines': {
                        'slop': 10,
                        'query': new_query
                    }
                }},
                {'match_phrase': {   # noqa: E122
                    'first_line': {
                        'slop': 10,
                        'query': new_query
                    }
                }}]
        body['query']['bool']['should'].extend(alt_clauses)

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        passage_similarity_long_lines(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim_use'], reverse=True)
    hits = hits[:5]
    return hits


@MemoizeQuery
def with_best_corpus_compounds_at_5_only_phrase_search(es, query, rerank=True):
    """Add using compounds computed from query dataset.

    Best submission on 5-June, improves a few qureies, NDCG - 0.31643
    """
    to_decompound, to_compound = most_freq_compound_corpus_strategy[0], most_freq_compound_corpus_strategy[1]
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
                    'remaining_lines': {
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

    new_query = to_compound_query(query, to_decompound, to_compound)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = \
                [{'match_phrase': {   # noqa: E127
                    'remaining_lines': {
                        'slop': 10,
                        'query': new_query
                    }
                }},
                {'match_phrase': {   # noqa: E122
                    'first_line': {
                        'slop': 10,
                        'query': new_query
                    }
                }}]
        body['query']['bool']['should'].extend(alt_clauses)

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity_long_lines(query, hit, verbose=False)

    if rerank:
        hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
        hits = hits[:5]
    return hits


@MemoizeQuery
def with_best_query_and_corpus_compounds_at_5_only_phrase_search(es, query, rerank=True):
    """Add using compounds computed from query dataset."""
    to_decompound, to_compound = most_freq_compound_both_strategy[0], most_freq_compound_both_strategy[1]
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
                    'remaining_lines': {
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

    new_query = to_compound_query(query, to_decompound, to_compound)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = \
                [{'match_phrase': {   # noqa: E127
                    'remaining_lines': {
                        'slop': 10,
                        'query': new_query
                    }
                }},
                {'match_phrase': {   # noqa: E122
                    'first_line': {
                        'slop': 10,
                        'query': new_query
                    }
                }}]
        body['query']['bool']['should'].extend(alt_clauses)

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        passage_similarity_long_lines(query, hit, verbose=False)

    if rerank:
        hits = sorted(hits, key=lambda x: x['_source']['max_sim_use'], reverse=True)
        hits = hits[:5]
    return hits


@MemoizeQuery
def with_best_compounds_at_5_only_first_line_use(es, query, rerank=True):
    """Shot in the dark on multiplying the USE score."""
    to_decompound, to_compound = most_freq_compound_strategy[0], most_freq_compound_strategy[1]
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
                    'remaining_lines': {
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

    new_query = to_compound_query(query, to_decompound, to_compound)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = \
                [{'match_phrase': {   # noqa: E127
                    'remaining_lines': {
                        'slop': 10,
                        'query': new_query
                    }
                }},
                {'match_phrase': {   # noqa: E122
                    'first_line': {
                        'slop': 10,
                        'query': new_query
                    }
                }}]
        body['query']['bool']['should'].extend(alt_clauses)

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity_long_lines(query, hit, verbose=False, remaining_lines=False)

    if rerank:
        hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
        hits = hits[:5]
    return hits


@MemoizeQuery
def with_best_compounds_at_5(es, query, rerank=True):
    """Add using compounds computed from query dataset."""
    to_decompound, to_compound = most_freq_compound_strategy[0], most_freq_compound_strategy[1]
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
                    'remaining_lines': {
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

    new_query = to_compound_query(query, to_decompound, to_compound)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = \
                [{'match_phrase': {   # noqa: E127
                    'remaining_lines': {
                        'slop': 10,
                        'query': new_query
                    }
                }},
                {'match_phrase': {   # noqa: E122
                    'first_line': {
                        'slop': 10,
                        'query': new_query
                    }
                }},
                {'match': {  # noqa: E122
                    'remaining_lines': {
                        'query': new_query
                    }
                }},
                {'match': {   # noqa: E122
                    'first_line': {
                        'query': new_query
                    }
                }},
                ]
        body['query']['bool']['should'].extend(alt_clauses)

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity_long_lines(query, hit, verbose=False)

    if rerank:
        hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
        hits = hits[:5]
    return hits


def with_best_compounds_at_20(es, query, rerank=False):
    """Add using compounds computed from query dataset.

    NDCG - 0.28790
    """
    to_decompound, to_compound = most_freq_compound_strategy[0], most_freq_compound_strategy[1]
    body = {
        'size': 20,
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
                    'remaining_lines': {
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

    new_query = to_compound_query(query, to_decompound, to_compound)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = [
            {
                'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': new_query
                    }
                }
            },
            {
                'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': new_query
                    }
                }
            },
            {'match': {
                'remaining_lines': {
                    'query': new_query
                }
            }},
            {'match': {
                'first_line': {
                    'query': new_query
                }
            }}
        ]
        body['query']['bool']['should'].extend(alt_clauses)

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity_long_lines(query, hit, verbose=False)

    if rerank:
        hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
        hits = hits[:5]

    return hits
