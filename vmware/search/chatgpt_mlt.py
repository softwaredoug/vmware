import json
# from .passage_similarity import passage_similarity_long_lines


def read_questions(filename='query_database.1.json'):
    query_database = json.load(open(filename, 'r'))
    return query_database['questions']


query_database = read_questions()


def chatgpt_mlt(es, query):
    article = query_database[query]
    lines = article.split('\n')

    first_line = lines[0]
    remaining_lines = '\n'.join(lines[1:])

    lowercase_first_line = first_line.lower()
    lowercase_first_line = lowercase_first_line.replace('title:', '')
    lowercase_first_line = lowercase_first_line.replace('title', '')

    body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "more_like_this": {
                            "boost": 10.0,
                            "fields": [
                                "first_line",
                            ],
                            "like": lowercase_first_line,
                        }
                    },
                    {
                        "more_like_this": {
                            "fields": [
                                "raw_text"
                            ],
                            "like": remaining_lines
                        }
                    },
                ]
            }
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    # for hit in hits:
    #    hit['_source']['max_sim'], hit['_source']['sum_sim'] \
    #        = passage_similarity_long_lines(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:10]
    return hits
