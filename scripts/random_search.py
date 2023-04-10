from time import time
import pickle
import random
from collections import defaultdict
from elasticsearch import Elasticsearch
import concurrent.futures
import pandas as pd
import os


import sys

sys.path.insert(0, '.')

from vmware.search.chatgpt_mlt import chatgpt_mlt  # noqa: E402


def ensure_dir():
    os.makedirs('data/random_search', exist_ok=True)


ensure_dir()


class BestDocsPerQuery:
    """Track the best document per query as per semantic similarity."""

    def __init__(self):
        try:
            self.best_doc_per_query = pickle.load(open('data/random_search/best_doc_per_query.pkl', 'rb'))
        except FileNotFoundError:
            self.best_doc_per_query = defaultdict(lambda: {'score': 0.0, 'doc_id': '', 'params': {}})

    def add_doc(self, query_id, doc_id, score, params, strategy, query):
        if query_id not in self.best_doc_per_query:
            self.best_doc_per_query[query_id] = {'score': 0.0, 'doc_id': '', 'params': {},
                                                 'strategy': '', 'query': ''}
        if score > self.best_doc_per_query[query_id]['score']:
            self.best_doc_per_query[query_id]['score'] = score
            self.best_doc_per_query[query_id]['doc_id'] = doc_id
            self.best_doc_per_query[query_id]['params'] = params
            self.best_doc_per_query[query_id]['strategy'] = strategy
            self.best_doc_per_query[query_id]['query'] = query

    def save(self):
        pickle.dump(self.best_doc_per_query, open('data/random_search/best_doc_per_query.pkl', 'wb'))


class ParamsHistory:

    def __init__(self):
        self.history = []
        self.best_params = {}
        self.best_score = 0.0
        self.timestamp = str(time()).replace('.', '')

    def add_params(self, params, score, strategy):
        self.history.append({'params': params, 'score': score,
                             'strategy': strategy.__name__})
        as_df = pd.DataFrame(self.history)
        fname = f"data/random_search/param_history_{self.timestamp}.csv"
        print(f"Writting param history to {fname}")
        as_df.to_csv(fname)
        if score > self.best_score:
            self.best_score = score
            self.best_params = params


def build_valid_params(es, test_query, strategy):
    """Build a set of valid params for a given strategy."""
    params = strategy.params
    params_dict = {param: random.uniform(0.1, 100.0) for param in params}
    params_good = False
    tries = 0
    while not params_good:
        try:
            strategy(es, query=test_query, params=params_dict)
            params_good = True
            break
        except ValueError:
            params_dict = {param: random.uniform(0.1, 100.0) for param in params}
            if tries > 100:
                raise ValueError("Could not find valid params")
        tries += 1
    return params_dict


def execute_run(strategy, es, queries,
                best_doc_per_query,
                num_queries=100,
                at=5):
    params_dict = build_valid_params(es,
                                     test_query=queries[0]['Query'],
                                     strategy=strategy)
    curr_score = 0.0
    for idx, query in enumerate(queries):
        results = strategy(es, query=query['Query'], params=params_dict)

        query_score = 0.0
        for rank, result in enumerate(results):
            doc_score = result['_source']['max_sim']
            query_score += result['_source']['max_sim']
            best_doc_per_query.add_doc(query_id=query['QueryId'],
                                       query=query['Query'],
                                       doc_id=result['_source']['id'],
                                       score=doc_score,
                                       params=params_dict,
                                       strategy=strategy.__name__)
            if rank >= at - 1:
                query_score /= at
                curr_score += query_score
                break
        if idx >= num_queries - 1:
            curr_score /= (idx + 1)
            break
    return curr_score, params_dict


def random_search(strategy=chatgpt_mlt,
                  num_queries=100, at=5):
    best_doc_per_query = BestDocsPerQuery()
    queries = pd.read_csv('data/test.csv').to_dict(orient='records')
    es = Elasticsearch('http://localhost:9200', timeout=30, max_retries=10,
                       retry_on_status=True, retry_on_timeout=True)

    params_history = ParamsHistory()

    concurrent_runs = 4
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_runs) as executor:
            futures = []
            for _ in range(0, concurrent_runs):
                futures.append(executor.submit(execute_run, strategy, es, queries,
                                               best_doc_per_query, num_queries, at))
            for future in concurrent.futures.as_completed(futures):
                curr_score, params_dict = future.result()
                print("----------")
                print(f"LAST SCORE      {curr_score} with params {params_dict}")
                print("----------")
                print(f"CURR BEST SCORE {params_history.best_score} with params {params_history.best_params}")
                best_doc_per_query.save()
                params_history.add_params(params_dict, curr_score, strategy)

    return params_history, best_doc_per_query


if __name__ == '__main__':
    params_history, best_doc_per_query = random_search()
    print("----------")
    print(f"FINAL BEST SCORE {params_history.best_score} with params {params_history.test_params}")
