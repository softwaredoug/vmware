from .passage_similarity import passage_similarity_long_lines
from .splainer import splainer_url
from .query_cache import MemoizeQuery
import json

from vmware.search.compounds import most_freq_compound_strategy, to_compound_query, \
    most_freq_compound_corpus_strategy, most_freq_compound_both_strategy


@MemoizeQuery
def with_best_compounds_at_50_plus_10_times_use(es, query, params=None, rerank=True):
    """Shot in the dark on multiplying the USE score differently.

    adding 10* to BM25 instead of replacing BM25 score. Best on 5 - June NDCG 0.31643
    """
    to_decompound, to_compound = most_freq_compound_strategy[0], most_freq_compound_strategy[1]
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

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        passage_similarity_long_lines(query, hit, verbose=False)

    if rerank:
        hits = sorted(hits, key=lambda x: ((10 * x['_source']['max_sim_use']) + x['_score']), reverse=True)
        hits = hits[:5]
    return hits


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

    print(json.dumps(body, indent=2))

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

    print(json.dumps(body, indent=2))

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

    print(json.dumps(body, indent=2))

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

    print(json.dumps(body, indent=2))

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

    print(json.dumps(body, indent=2))

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

    new_query = []
    last_term = ''
    fast_forward = False
    for first_term, second_term in zip(query.split(), query.split()[1:]):
        last_term = second_term
        if fast_forward:
            print("Skipping: " + first_term + " " + second_term)
            fast_forward = False
            continue
        first_term = first_term.strip().lower()
        second_term = second_term.strip().lower()
        if first_term in to_decompound:
            new_query.append(to_decompound[first_term])
        elif (first_term, second_term) in to_compound:
            new_query.append(first_term + second_term)
            fast_forward = True
        else:
            new_query.append(first_term)

    if last_term in to_decompound:
        new_query.append(to_decompound[last_term])
    else:
        new_query.append(last_term)

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

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity_long_lines(query, hit, verbose=False)

    if rerank:
        hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
        hits = hits[:5]

    return hits
