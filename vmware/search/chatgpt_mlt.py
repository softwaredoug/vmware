import json
from collections import defaultdict
from .passage_similarity import passage_similarity_long_lines
from .splainer import splainer_url


def strip_newlines_from_keys(d):
    return {k.replace('\n', ''): v for k, v in d.items()}


def read_questions():
    questions = defaultdict(dict)
    for idx in range(1, 100):
        filename = f'gpt_articles/query_database.{idx}.json'
        print("Reading:", filename)
        try:
            with open(filename, 'r') as f:
                query_database = json.load(f)
                question_articles = strip_newlines_from_keys(query_database['questions'])
                for query, article in question_articles.items():
                    lines = article.split('\n')
                    first_line = lines[0]
                    lowercase_first_line = first_line.lower()
                    lowercase_first_line = lowercase_first_line.replace('title:', '')
                    lowercase_first_line = lowercase_first_line.replace('question:', '')
                    lowercase_first_line = lowercase_first_line.replace('title', '')
                    try:
                        questions[query]['first_line'].append(lowercase_first_line)
                        questions[query]['raw_text'].append(article)
                    except KeyError:
                        questions[query]['first_line'] = [lowercase_first_line]
                        questions[query]['raw_text'] = [article]
        except FileNotFoundError:
            return questions


query_database = read_questions()


def chatgpt_mlt(es, query, params):

    try:
        first_lines = " | ".join(query_database[query]['first_line'])
        article = " | ".join(query_database[query]['raw_text'])
    except KeyError:
        first_lines = query
        article = query

    body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "more_like_this": {
                            "boost": params['title_mlt_boost'],
                            "fields": [
                                "first_line",
                            ],
                            "like": first_lines,
                        }
                    },
                    {
                        "more_like_this": {
                            "boost": params['body_mlt_boost'],
                            "fields": [
                                "raw_text"
                            ],
                            "like": article,
                        }
                    },
                ]
            }
        }
    }

    # print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']
    for hit in hits:
        hit['_source']['splainer'] = splainer_url(es_body=body)

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] \
            = passage_similarity_long_lines(query, hit, verbose=False)
    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:10]
    return hits
