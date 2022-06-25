import pandas as pd
import numpy as np
import itertools
from vmware.simulate.diff_simulation import estimate_relevance, ideal_dcg_at_5, create_results_diff, likelihood_not_random


def significant_diffs(results):
    diffs = []
    results_by_significance = []
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

    diffs = sorted(diffs, key=lambda x: x['dcg_delta_per_changed'], reverse=True)
    for diff in diffs:
        results_by_significance.append((diff['before'], diff['results_before'], diff['ndcg_before']))
        results_by_significance.append((diff['after'], diff['results_after'], diff['ndcg_after']))

    return results_by_significance


def grade_judgments(judgments):
    # judgments['alpha'] *= judgments['dcg_delta_per_changed']
    # judgments['beta'] *= judgments['dcg_delta_per_changed']
    judgments = \
        judgments.groupby(['QueryId', 'DocumentId'])[['alpha', 'beta']].sum()
    judgments['grade'] = judgments['alpha'] / (judgments['alpha'] + judgments['beta'])
    judgments['grade_std_dev'] = np.sqrt((judgments['alpha'] * judgments['beta']) /
                                         (((judgments['alpha'] + judgments['beta'])**2) * (1 + judgments['alpha'] + judgments['beta'])))
    return judgments


def run_diffs(results):
    judgments = pd.DataFrame(columns=['QueryId', 'DocumentId', 'alpha', 'beta',
                                      'weight_delta', 'position_delta'])

    all_diffs = []

    for result_before, result_after in zip(results, results[1:]):
        name_before, results_before, ndcg_before = result_before
        name_after, results_after, ndcg_after = result_after

        mean_ndcg_diff = ndcg_after - ndcg_before

        if mean_ndcg_diff <= 0:
            continue

        results_diff = create_results_diff(results_before, results_after)
        results_diff['name_before'] = name_before
        results_diff['name_after'] = name_after

        # Translate our NDCG@5 to a sum of all DCG@5 to simplify the simulation
        actual_dcg_delta = len(results_diff['QueryId'].unique()) * mean_ndcg_diff * ideal_dcg_at_5

        results = estimate_relevance(results_diff,
                                     actual_dcg_delta=actual_dcg_delta,
                                     verbose=True)

        # Accumulate judgments from this pair into the evaluation
        assert (results[results['position_delta'] == 0]['weight_delta'] == 0).all()
        all_diffs.append(results)

    judgments = pd.concat(all_diffs)
    judgments.to_pickle('data/judgments.pkl')
    return judgments


def to_submission(results, name):
    from time import time
    timestamp = str(time()).replace('.', '')
    fname = f'data/simulated_{name}_turnbull_{timestamp}.csv'
    results = results.sort_values(['QueryId', 'grade'], ascending=[True, False]).groupby('QueryId').head(5)
    print("Writing To: ", fname)
    results[['QueryId', 'DocumentId']].to_csv(fname, index=False)
