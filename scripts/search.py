# from . import path  # noqa: F401
from sys import argv
import sys
from elasticsearch import Elasticsearch
from collections import defaultdict
import pickle
import random
from math import log

import pandas as pd
# Best baseline thusfar
# from .rerank_simple_slop_search import \
#    rerank_slop_search_remaining_lines_max_snippet_at_5
# from vmware.search.compound_search import with_best_compounds_at_5_only_phrase_search
sys.path.insert(0, '.')
from vmware.search.chatgpt_mlt import chatgpt_mlt  # noqa: E402
from vmware.search.rerank_simple_slop_search import \
    rerank_simple_slop_search_max_snippet_at_5, rerank_simple_slop_search  # noqa: E402, F401


def damage(results1, results2, at=10):
    """How "damaging" could the change from results1 -> results2 be.

    (results1, results2 are an array of document identifiers)
    For each result in result1,
        Is the result in result2[:at]
            If so, how far has it moved?
        If not,
            Consider it a move of at+1
        damage += discount(idx) * moveDist
    """

    def discount(idx):
        return 1.0 / log(idx + 2)

    idx = 0
    dmg = 0.0

    if len(results1) < at:
        at = len(results1)

    for result in results1[:at]:
        movedToIdx = at + 1  # out of the window
        if result in results2:
            movedToIdx = results2.index(result)
        moveDist = abs(movedToIdx - idx)
        dmg += discount(idx) * moveDist
        idx += 1

    return dmg


def search(query,
           strategy=rerank_simple_slop_search_max_snippet_at_5):
    print(query)
    es = Elasticsearch('http://localhost:9200', timeout=30, max_retries=10,
                       retry_on_status=True, retry_on_timeout=True)
    hits = strategy(es, query=query)
    for hit in hits:
        print("**********************************")
        print(hit['_source']['title'] if 'title' in hit['_source'] else '',
              '||',
              hit['_source']['first_line'])
        max_sim = 'NA'
        if 'max_sim' in hit['_source']:
            max_sim = hit['_source']['max_sim']
        print(f"MAX SIM {max_sim} | SCORE {hit['_score']}")
        print("----------------------------------")


def submission(strategy=chatgpt_mlt):
    queries = pd.read_csv('data/test.csv')
    all_results = []
    es = Elasticsearch('http://localhost:9200', timeout=30, max_retries=10,
                       retry_on_status=True, retry_on_timeout=True)
    for query in queries.to_dict(orient='records'):
        results = strategy(es, query=query['Query'])
        for rank, result in enumerate(results):
            query_result = {}
            query_result['QueryId'] = query['QueryId']
            query_result['rank'] = rank
            query_result['DocumentId'] = result['_source']['id']
            all_results.append(query_result)

            if rank >= 4:
                assert len(all_results) % 5 == 0
                break

    df = pd.DataFrame(all_results)
    write_submission(df, strategy.__name__)


def random_search(strategy=chatgpt_mlt,
                  num_queries=100, at=5, runs=40):
    max_score = 0.0
    params = strategy.params
    best_params = {param: -1.0 for param in params}
    try:
        best_doc_per_query = pickle.load(open('best_doc_per_query.pkl', 'rb'))
    except FileNotFoundError:
        print("RESTARTING best docs")
        best_doc_per_query = defaultdict(lambda: {'score': 0.0, 'doc_id': '', 'params': {}})
    queries = pd.read_csv('data/test.csv')
    es = Elasticsearch('http://localhost:9200', timeout=30, max_retries=10,
                       retry_on_status=True, retry_on_timeout=True)
    param_history = []
    for i in range(0, runs):
        params_dict = {param: random.uniform(0.1, 100.0) for param in params}
        params_good = False

        curr_score = 0.0
        for idx, query in enumerate(queries.to_dict(orient='records')):

            # Ensure valid params
            tries = 0
            results = None
            while not params_good:
                try:
                    results = strategy(es, query=query['Query'], params=params_dict)
                    params_good = True
                    break
                except ValueError:
                    params_dict = {param: random.uniform(0.1, 100.0) for param in params}
                    if tries > 100:
                        raise ValueError("Could not find valid params")
                tries += 1
            results = strategy(es, query=query['Query'], params=params_dict)

            query_score = 0.0
            for rank, result in enumerate(results):
                doc_score = result['_source']['max_sim']
                query_score += result['_source']['max_sim']
                if doc_score > best_doc_per_query[query['QueryId']]['score']:
                    best_doc_per_query[query['QueryId']]['score'] = result['_source']['max_sim']
                    best_doc_per_query[query['QueryId']]['doc_id'] = result['_source']['id']
                    best_doc_per_query[query['QueryId']]['params'] = params_dict
                    best_doc_per_query[query['QueryId']]['strategy'] = strategy.__name__
                    best_doc_per_query[query['QueryId']]['query'] = query['Query']
                if rank >= at - 1:
                    query_score /= at
                    curr_score += query_score
                    break
            if idx >= num_queries:
                curr_score /= (idx + 1)
                print(idx, query['Query'], best_doc_per_query[query['QueryId']])
                break
        if curr_score > max_score:
            max_score = curr_score
            best_params = params_dict
        print("----------")
        print("----------")
        print(f"CURR BEST SCORE {max_score} with params {best_params}")
        pickle.dump(dict(best_doc_per_query), open('best_doc_per_query.pkl', 'wb'))
        param_history.append({'params': params_dict, 'score': curr_score,
                              'strategy': strategy.__name__})
        as_df = pd.DataFrame(param_history)
        as_df.to_csv('param_history.csv', mode='a', header=False)
    return param_history, best_doc_per_query


def debug(baseline=rerank_simple_slop_search_max_snippet_at_5,
          test=chatgpt_mlt,
          verbose=False):
    """Search all test queries to generate a submission."""
    queries = pd.read_csv('data/test.csv')
    all_results = []
    es = Elasticsearch('http://localhost:9200', timeout=30, max_retries=10,
                       retry_on_status=True, retry_on_timeout=True)
    for query in queries.to_dict(orient='records'):
        results_control = baseline(es, query=query['Query'])
        results_test = test(es, query=query['Query'])

        delta_damage = damage([r['_id'] for r in results_control],
                              [r['_id'] for r in results_test])

        if verbose:
            print(query['Query'])
            print(f"DAMAGE: {delta_damage}")
            print("----------------------------------")

        for rank, (result_control, result_test) in enumerate(zip(results_control, results_test)):
            source_control = result_control['_source']
            source_test = result_test['_source']
            source_test['rank'] = rank
            source_test['score_test'] = result_test['_score']
            source_test['score_control'] = result_control['_score']
            source_test['damage'] = delta_damage
            source_test['DocumentId_test'] = source_test['id']
            source_test['DocumentId'] = source_test['id']
            source_test['DocumentId_control'] = source_control['id']
            source_test['splainer_test'] = source_test['splainer']
            source_test['splainer_control'] = source_control['splainer']
            source_test['first_line_control'] = source_control['first_line']
            source_test['first_line_test'] = source_test['first_line']
            source_test['raw_text_control'] = source_control['raw_text']
            source_test['QueryId'] = query['QueryId']
            all_results.append(source_test)
    all_results = pd.DataFrame(all_results)
    all_results = queries.merge(all_results, how='left', on='QueryId')\
        .sort_values(['QueryId', 'rank'])
    write_submission(all_results, test.__name__)
    return all_results


def write_submission(all_results, name):
    from time import time
    timestamp = str(time()).replace('.', '')
    fname = f'data/{name}_turnbull_{timestamp}.csv'
    print("Writing To: ", fname)
    all_results[['QueryId', 'DocumentId']].to_csv(fname, index=False)


def diff_results(all_results):
    diff_queries = all_results[all_results['damage'] > 0][['Query', 'damage',
                                                           'first_line_control', 'first_line',
                                                           'splainer_test', 'splainer_control']]
    last_query = None
    for result in diff_queries.sort_values(['damage', 'Query']).to_dict(orient='records'):
        if result['Query'] != last_query:
            last_query = result['Query']
            print(f"-------- {result['Query']} -- {result['damage']}-------")
        print(result['first_line_control'], "|||", result['first_line'])
    print("----------------------------------")
    print(f"Changed Queries - {len(diff_queries['Query'].unique())} different queries")


if __name__ == "__main__":
    search(argv[1], strategy=chatgpt_mlt)
