import json
from operator import itemgetter
from collections import defaultdict
from .passage_similarity import passage_similarity_long_lines
from .splainer import splainer_url


def strip_newlines_from_keys(d):
    return {k.replace('\n', ''): v for k, v in d.items()}


def read_gpt_file(filename):
    with open(filename, 'r') as f:
        print("Reading:", filename)
        query_database = json.load(f)
        question_articles = strip_newlines_from_keys(query_database['questions'])
        for query, article in question_articles.items():
            if isinstance(article, dict):
                article = article['article']
            question_articles[query] = article
        return question_articles


def expansions():
    queries = {}
    for idx in range(1, 100):
        filename = f'gpt_articles/expansions.{idx}.json'
        try:
            this_questions = read_gpt_file(filename)
            for query, article in this_questions.items():
                query_expansions = []
                if query not in queries:
                    queries[query] = []
                lines = article.split('\n')
                for line in lines:
                    if line.startswith('*'):
                        query_expansions.append(line[1:])

                queries[query] = query_expansions

        except FileNotFoundError:
            break
    return queries


def read_questions():
    questions = defaultdict(dict)
    num_files = 0
    for idx in range(1, 100):
        filename = f'gpt_articles/query_database.{idx}.json'
        try:
            with open(filename, 'r') as f:
                print("Reading:", filename)
                query_database = json.load(f)
                question_articles = strip_newlines_from_keys(query_database['questions'])
                for query, article in question_articles.items():
                    if query not in questions:
                        questions[query] = {'first_line': [],
                                            'raw_text': []}
            num_files = idx
        except FileNotFoundError:
            break

    for query in questions.keys():
        questions[query]['first_line'] = [''] * num_files
        questions[query]['raw_text'] = [''] * num_files

    for idx in range(1, 100):
        filename = f'gpt_articles/query_database.{idx}.json'
        try:
            this_questions = read_gpt_file(filename)
            for query, article in this_questions.items():
                lines = article.split('\n')
                first_line = lines[0]
                lowercase_first_line = first_line.lower()
                lowercase_first_line = lowercase_first_line.replace('title:', '')
                lowercase_first_line = lowercase_first_line.replace('question:', '')
                lowercase_first_line = lowercase_first_line.replace('title', '')

                questions[query]['first_line'][idx - 1] = lowercase_first_line
                questions[query]['raw_text'][idx - 1] = article
        except FileNotFoundError:
            return questions


query_database = read_questions()
expansions = expansions()


def better_item_getter(indices, src):
    if len(indices) == 0:
        raise ValueError("No indices to get")
    if len(indices) == 1:
        return [src[indices[0]]]
    else:
        return list(itemgetter(*indices)(src))


def chatgpt_mlt(es, query, params):

    try:
        items = query_database[query]['first_line']
        item_indices = []
        for idx, item in enumerate(items):
            if params[f"use_{idx+1}"] > 50.0:
                item_indices.append(idx)

        if len(item_indices) == 0:
            raise ValueError("No items to use")

        assert len(query_database[query]['first_line']) == len(query_database[query]['raw_text'])
        assert len(query_database[query]['first_line']) == 5

        first_lines_to_use = better_item_getter(item_indices, query_database[query]['first_line'])
        raw_text_to_use = better_item_getter(item_indices, query_database[query]['raw_text'])
        first_lines = ' | ' .join(first_lines_to_use)
        article = ' | ' .join(raw_text_to_use)

        if params['use_query'] > 50.0:
            first_lines = query + ' | ' + first_lines
            article = query + ' | ' + article
    except KeyError:
        first_lines = query
        article = query

    try:
        expansion = " OR ".join(expansions[query])
    except KeyError:
        expansion = ""

    rerank_depth = int(params['rerank_depth'])
    if rerank_depth < 5:
        raise ValueError("Rerank depth must be at least 5")

    for mlt_param in ['title_mlt_min_term_freq', 'title_mlt_min_word_length', 'title_mlt_min_doc_freq', 'title_mlt_max_query_terms',
                      'body_mlt_min_term_freq', 'body_mlt_min_word_length', 'body_mlt_min_doc_freq', 'body_mlt_max_query_terms']:
        if int(params[mlt_param]) < 1:
            raise ValueError(f"{mlt_param} must be at least 1")

    body = {
        "size": rerank_depth,
        "query": {
            "bool": {
                "should": [
                    {'match_phrase': {
                        'remaining_lines': {
                            'slop': int(params['remaining_lines_slop']),
                            'query': query,
                            'boost': float(params['remaining_lines_phrase_boost'])
                        }
                    }},
                    {'match_phrase': {
                        'first_line': {
                            'slop': int(params['first_line_slop']),
                            'query': query,
                            'boost': float(params['first_line_phrase_boost'])
                        }
                    }},
                    {'query_string': {
                        'query': expansion,
                        'boost': float(params['body_expansions_boost']),
                        'fields': ['raw_text']
                    }},
                    {'query_string': {
                        'query': expansion,
                        'boost': float(params['title_expansions_boost']),
                        'fields': ['first_line']
                    }},
                    {'match': {
                        'raw_text': {
                            'query': query,
                            'boost': float(params['raw_text_boost'])
                        }
                    }},
                    {'match': {
                        'first_line': {
                            'query': query,
                            'boost': float(params['first_line_boost'])
                        }
                    }},
                    {
                        "more_like_this": {
                            "boost": params['title_mlt_boost'],
                            "fields": [
                                "first_line",
                            ],
                            "like": first_lines,
                            "min_term_freq":int(params['title_mlt_min_term_freq']),
                            "min_word_length":int(params['title_mlt_min_word_length']),
                            "min_doc_freq":int(params['title_mlt_min_doc_freq']),
                            "max_query_terms":int(params['title_mlt_max_query_terms'])
                        }
                    },
                    {
                        "more_like_this": {
                            "boost": params['body_mlt_boost'],
                            "fields": [
                                "raw_text"
                            ],
                            "like": article,
                            "min_term_freq":int(params['body_mlt_min_term_freq']),
                            "min_word_length":int(params['body_mlt_min_word_length']),
                            "min_doc_freq":int(params['body_mlt_min_doc_freq']),
                            "max_query_terms":int(params['body_mlt_max_query_terms'])
                        }
                    },
                ]
            }
        }
    }
    import pdb; pdb.set_trace()

    hits = es.search(index='vmware', body=body)['hits']['hits']
    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] \
            = passage_similarity_long_lines(query, hit, verbose=False)
    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:10]
    return hits


chatgpt_mlt.params = ['body_mlt_boost', 'title_mlt_boost',
                      'use_1', 'use_2', 'use_3', 'use_4', 'use_5', 'use_6', 'use_query',
                      'title_expansions_boost', 'body_expansions_boost',
                      'body_mlt_min_term_freq', 'body_mlt_min_word_length',
                      'body_mlt_min_doc_freq', 'body_mlt_max_query_terms',
                      'title_mlt_min_term_freq', 'title_mlt_min_word_length',
                      'title_mlt_min_doc_freq', 'title_mlt_max_query_terms',
                      'rerank_depth', 'first_line_slop', 'first_line_boost',
                      'first_line_phrase_boost', 'remaining_lines_slop',
                      'remaining_lines_phrase_boost', 'raw_text_boost']
