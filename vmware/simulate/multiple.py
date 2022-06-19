import pandas as pd
import numpy as np
import itertools
from vmware.simulate.diff_simulation import estimate_relevance as single_diff_relevance, ideal_dcg_at_5, create_results_diff, likelihood_not_random


def significant_diffs(results):
    diffs = []
    for result1, result2 in itertools.combinations(results, 2):
        name_before, results_before, ndcg_before = result1
        name_after, results_after, ndcg_after = result2
        if ndcg_before < ndcg_after:
            results_diff = create_results_diff(results_before, results_after)
            mean_ndcg_diff = ndcg_after - ndcg_before
            actual_dcg_delta = len(results_diff['QueryId'].unique()) * mean_ndcg_diff * ideal_dcg_at_5
            not_random = likelihood_not_random(results_diff, actual_dcg_delta)
            print(name_before, name_after, not_random)
            diffs.append(({'before': name_before,
                           'after': name_after,
                           'num_changed': len(results_diff.loc[results_diff['weight_delta'] != 0]),
                           'not_random': not_random,
                           'actual_dcg_delta': actual_dcg_delta,
                           'best_dcg_delta': results_diff['best_case_dcg_delta'].iloc[0]}))
    return pd.DataFrame(diffs).sort_values(by='not_random', ascending=False)


def estimate_relevance(results):
    judgments = pd.DataFrame(columns=['QueryId', 'DocumentId', 'alpha', 'beta',
                                      'weight_delta', 'position_delta'])

    results = sorted(results, key=lambda x: x[2])
    for result_before, result_after in zip(results, results[1:]):
        name_before, results_before, ndcg_before = result_before
        name_after, results_after, ndcg_after = result_after

        mean_ndcg_diff = ndcg_after - ndcg_before

        results_diff = create_results_diff(results_before, results_after)

        # Translate our NDCG@5 to a sum of all DCG@5 to simplify the simulation
        actual_dcg_delta = len(results_diff['QueryId'].unique()) * mean_ndcg_diff * ideal_dcg_at_5

        results = single_diff_relevance(results_diff,
                                        actual_dcg_delta=actual_dcg_delta,
                                        verbose=True)

        # Accumulate judgments from this pair into the evaluation
        assert (results[results['position_delta'] == 0]['weight_delta'] == 0).all()
        prob_not_random = results['prob_not_random'].iloc[0]
        results = results.groupby(['QueryId', 'DocumentId'])[['alpha', 'beta']].sum()
        # Score on how relevant we think this simulation is to the overall alpha / beta
        results['alpha'] *= prob_not_random
        results['beta'] *= prob_not_random
        if len(judgments) == 0:
            judgments = results
        else:
            judgments = judgments.append(results)

        judgments = \
            judgments.groupby(['QueryId', 'DocumentId'])[['alpha', 'beta']].sum()
        print(len(results), '->', len(judgments))
    judgments['grade'] = judgments['alpha'] / (judgments['alpha'] + judgments['beta'])
    judgments['grade_std_dev'] = np.sqrt((judgments['alpha'] * judgments['beta']) /
                                         (((judgments['alpha'] + judgments['beta'])**2) * (1 + judgments['alpha'] + judgments['beta'])))
    return judgments
