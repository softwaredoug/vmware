from time import time
import pickle
import random
from collections import defaultdict
from elasticsearch import Elasticsearch
import concurrent.futures
import pandas as pd
import os
import glob

import sys

sys.path.insert(0, '.')

from vmware.search.chatgpt_mlt import chatgpt_mlt  # noqa: E402, F401
from vmware.search.rerank_simple_slop_search import rerank_simple_slop_search  # noqa: E402, F401
from vmware.search.compound_search import with_best_compounds_at_50_plus_10_times_use


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
        pickle.dump(dict(self.best_doc_per_query), open('data/random_search/best_doc_per_query.pkl', 'wb'))


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


def build_valid_params(es, test_query, strategy, hard_coded):
    """Build a set of valid params for a given strategy."""
    params = strategy.params
    params_dict = {param: random.uniform(0.1, 100.0) for param in params}
    for key, value in hard_coded.items():
        params_dict[key] = value
    params_good = False
    tries = 0
    while not params_good:
        try:
            strategy(es, query=test_query, params=params_dict)
            params_good = True
            break
        except ValueError:
            params_dict = {param: random.uniform(0.1, 100.0) for param in params}
            for key, value in hard_coded.items():
                params_dict[key] = value
            if tries > 100:
                raise ValueError("Could not find valid params")
        tries += 1
    return params_dict


def execute_run(strategy, es, queries,
                best_doc_per_query,
                num_queries=500,
                hard_coded={},
                at=5):
    params_dict = build_valid_params(es,
                                     test_query=queries[0]['Query'],
                                     strategy=strategy,
                                     hard_coded=hard_coded)
    curr_score = 0.0
    for idx, query in enumerate(queries):
        results = strategy(es, query=query['Query'], params=params_dict)

        query_score = 0.0
        for rank, result in enumerate(results):
            doc_score = result['_source']['max_sim_mpnet']
            query_score += doc_score
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
                  num_queries=100, at=5,
                  hard_coded={}):
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
                                               best_doc_per_query, num_queries, hard_coded, at))
            for future in concurrent.futures.as_completed(futures):
                curr_score, params_dict = future.result()
                params_history.add_params(params_dict, curr_score, strategy)
                best_doc_per_query.save()
                print("----------")
                print(f"LAST SCORE      {curr_score} with params {params_dict}")
                print("----------")
                print(f"CURR BEST SCORE {params_history.best_score} with params {params_history.best_params}")

    return params_history, best_doc_per_query


def read_all_csv(path='data/random_search'):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    return pd.concat(df_from_each_file, ignore_index=True)


hardcoded = {'use_1': 51.206125364348992,
             'use_2': 51.685299030133455,
             'use_3': 51.07842982196352,
             'use_4': 51.53141784404721,
             'use_5': 51.70771331344108,
             'use_6': 51.70771331344108,
             'use_query': 51.9313524558668638,
             'rerank_depth': 100}


if __name__ == '__main__':
    print("Running random search, stop with Ctrl+C")
    params_history, best_doc_per_query = random_search(strategy=with_best_compounds_at_50_plus_10_times_use,

                                                       hard_coded={})
    print("----------")
    print(f"FINAL BEST SCORE {params_history.best_score} with params {params_history.test_params}")
