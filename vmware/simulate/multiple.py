import pandas as pd
import numpy as np
import itertools
from time import time
import os.path
from vmware.simulate.diff_simulation import estimate_relevance, ideal_dcg_at_5, create_results_diff, likelihood_not_random

from concurrent.futures import ProcessPoolExecutor, as_completed


def significant_diffs(results):
    diffs = []
    results_by_significance = []
    # This doesn't work for going from no results -> results because num changed is so high
    for result1, result2 in itertools.combinations(results, 2):
        name_before, results_before, ndcg_before = result1
        name_after, results_after, ndcg_after = result2
        if ndcg_before < ndcg_after:
            results_diff = create_results_diff(results_before, results_after)
            mean_ndcg_diff = ndcg_after - ndcg_before
            actual_dcg_delta = len(results_diff['QueryId'].unique()) * mean_ndcg_diff * ideal_dcg_at_5
            not_random = likelihood_not_random(results_diff, actual_dcg_delta)
            num_changed = len(results_diff.loc[results_diff['weight_delta'] != 0])
            print(name_before, name_after, not_random)
            diffs.append(({'before': name_before,
                           'results_before': results_before,
                           'ndcg_before': ndcg_before,
                           'after': name_after,
                           'results_after': results_after,
                           'ndcg_after': ndcg_after,
                           'num_changed': num_changed,
                           'ndcg_diff': mean_ndcg_diff,
                           'not_random': not_random,
                           'dcg_delta_per_changed': actual_dcg_delta / num_changed,
                           'actual_dcg_delta': actual_dcg_delta,
                           'best_dcg_delta': results_diff['best_case_dcg_delta'].iloc[0]}))

    # diffs = sorted(diffs, key=lambda x: x['not_random'], reverse=True)
    print("------------------------")
    for diff in diffs:
        print(f"{diff['before']}->{diff['after']} | {diff['not_random']:.3f}")
    for diff in diffs:
        results_by_significance.append((diff['before'], diff['results_before'], diff['ndcg_before']))
        results_by_significance.append((diff['after'], diff['results_after'], diff['ndcg_after']))

    return results_by_significance


def information_amount(all_diffs, midpoint=0.6, skew=1000000):
    """Scale the rels and not_rels to the available information. Diff close to 1 perfect info. Close to 0, very
       very imperfect info."""
    # See graph here https://www.desmos.com/calculator/0wah7odcqh
    deltas = np.abs(all_diffs['rels'] - all_diffs['not_rels'])
    return skew ** (deltas - midpoint)


def grade_judgments(all_diffs):
    all_diffs['weight'] = information_amount(all_diffs)

    all_diffs['rels'] *= all_diffs['weight']
    all_diffs['not_rels'] *= all_diffs['weight']
    judgments = \
        all_diffs.groupby(['QueryId', 'DocumentId'])[['rels', 'not_rels']].sum()
    judgments['grade'] = judgments['rels'] / (judgments['rels'] + judgments['not_rels'])
    judgments['grade_std_dev'] = np.sqrt((judgments['rels'] * judgments['not_rels']) /
                                         (((judgments['rels'] + judgments['not_rels'])**2) * (1 + judgments['rels'] + judgments['not_rels'])))
    return judgments


def debugable_results(judgments):
    # Join with corpus for debugging
    corpus = pd.read_csv('data/vmware_ir_content.csv')
    queries = pd.read_csv('data/test.csv')
    judgments = judgments.reset_index()
    results = judgments.merge(corpus, right_on='f_name', left_on='DocumentId', how='left')
    results = results.merge(queries, on='QueryId', how='left')
    return results


def diff_worker(result_before, result_after):
    name_before, results_before, ndcg_before = result_before
    name_after, results_after, ndcg_after = result_after

    mean_ndcg_diff = ndcg_after - ndcg_before

    if mean_ndcg_diff <= 0:
        return

    if name_before is not None:
        cache_file = f"data/cache/{os.path.basename(name_before)}_{os.path.basename(name_after)}.pkl"
    else:
        cache_file = f"data/cache/{None}_{os.path.basename(name_after)}.pkl"

    try:
        results = pd.read_pickle(cache_file)
        print(f"Reusing Cache {cache_file}")
    except FileNotFoundError:
        results_diff = create_results_diff(results_before, results_after)
        results_diff['name_before'] = name_before
        results_diff['name_after'] = name_after

        # Translate our NDCG@5 to a sum of all DCG@5 to simplify the simulation
        actual_dcg_delta = len(results_diff['QueryId'].unique()) * mean_ndcg_diff * ideal_dcg_at_5

        results = estimate_relevance(results_diff,
                                     actual_dcg_delta=actual_dcg_delta,
                                     verbose=True)
        results.to_pickle(cache_file)
    assert (results[results['position_delta'] == 0]['weight_delta'] == 0).all()

    # Any changed results should have participated in the simulation, ignore others
    results = results[results['weight_delta'] != 0.0]
    assert (results['rels'] + results['not_rels'] > 0.95).all()
    return results


def run_diffs(results):
    all_diffs = []

    futures = []

    with ProcessPoolExecutor(max_workers=8) as executor:

        for result_before, result_after in zip(results, results[1:]):
            name_before, results_before, ndcg_before = result_before
            name_after, results_after, ndcg_after = result_after

            mean_ndcg_diff = ndcg_after - ndcg_before
            if mean_ndcg_diff > 0.0:
                future = executor.submit(diff_worker, result_before, result_after)
                futures.append(future)

    for future in as_completed(futures):
        results = future.result()
        all_diffs.append(results)

    all_diffs = pd.concat(all_diffs)
    all_diffs.to_pickle("data/all_diffs.pkl")
    return all_diffs


def to_submission(results, name):
    timestamp = str(time()).replace('.', '')
    fname = f'data/simulated_{name}_turnbull_{timestamp}.csv'
    results = results.sort_values(['QueryId', 'grade'], ascending=[True, False]).groupby('QueryId').head(5)
    print("Writing To: ", fname)
    results[['QueryId', 'DocumentId']].to_csv(fname, index=False)
