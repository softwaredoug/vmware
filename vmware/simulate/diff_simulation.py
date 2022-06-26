import pandas as pd
import numpy as np
from statistics import NormalDist
from time import perf_counter
from vmware.helpers import Memoize

ideal_dcg_at_5 = 2.948459119


def add_weights(diff):
    """Add a position. Compute DCG weights"""
    # Why does this give pandas warning? It is already using loc[]
    diff.loc[:, 'position'] = diff.groupby('QueryId').cumcount() + 1
    diff.loc[:, 'weight'] = 1 / np.log2(diff['position'] + 1)
    return diff


# Ideeal DCG@5 with weight 1 / log(n+1), if n is 1 based position
ideal_dcg_at_5 = 2.948459119
# one unit of NDCG == ideal_dcg units of DCG
#
# total DCG change =
#  NDCG * ideal_dcg * num_queries
#
#  0-1 total possible sales (1 highest possible)
#  ideal sales number
#
# what if this was an A/B test instead of NDCG?
#  - assume some weighted correlation of position ranking to A/B testing outcome
#  - we could somehow know those weights (ie best item in position 1 nets 5 sales. But in position 2 nets 3)
#  - given total sales between two rankings
#  -
#
# (note this -- imperfectly -- forces the NDCG from the source system into our DCG scaling)
#
# For every new result, with DCG position weight wn. Each result has gone from position wn -> wm
#
# For every moved search result with a relevance grade g, we can see the change in DCG as follows
#
#  delta_dcg = (wn - wm) * g + ...  (for every changed query/doc pair)
#
# Where we assume grade g is either 0 (irrelevant) or 1 (relevant)
#
# Perfect case, only permutation that creates this is when they're all 1
#
#  max_delta_dcg = sum(wn - wm) for all wn > wm
#
# When we detect an actuall delta_dcg, we can randomly select grades 0 and 1 to each doc to see
# which ones most approximate the delta dcg


def create_results_diff(results_before, results_after):
    """Compute the DCG delta weights resulting from the before and after,
       so we can compare to observed mean delta dcg."""
    results_after = results_after.groupby('QueryId').head(5)
    results_after = add_weights(results_after)

    if results_before is not None:
        results_before = results_before.groupby('QueryId').head(5)
        assert len(results_before) == len(results_after)
        results_before = add_weights(results_before)
        diff = results_before.merge(results_after,
                                    on=['QueryId', 'DocumentId'],
                                    how='outer')
        diff = diff.rename(
            columns={'position_x': 'position_before',
                     'position_y': 'position_after',
                     'weight_x': 'weight_before',
                     'weight_y': 'weight_after'}
        )
    else:
        diff = results_after
        diff['position_after'] = diff['position']
        diff['position_before'] = 0
        diff['weight_after'] = diff['weight']
        diff['weight_before'] = 0

    diff['weight_after'] = diff['weight_after'].replace(np.nan, 0)
    diff['weight_before'] = diff['weight_before'].replace(np.nan, 0)
    diff['weight_delta'] = diff['weight_after'] - diff['weight_before']
    diff['position_delta'] = diff['position_after'] - diff['position_before']
    diff['weight_delta_abs'] = np.abs(diff['weight_delta'])
    diff.loc[:, 'rels'] = 0.001
    diff.loc[:, 'not_rels'] = 0.001
    assert (diff[diff['position_delta'] == 0]['weight_delta'] == 0).all()

    diff.loc[:, 'grade'] = 0
    diff.loc[diff['weight_delta'] > 0, 'grade'] = 1
    diff.loc[diff['weight_delta'] < 0, 'grade'] = 0
    best_case_dcg_delta = sum(diff['grade'] * diff['weight_delta'])

    diff['best_case_dcg_delta'] = best_case_dcg_delta
    num_grades_changed = len(diff.loc[diff['weight_delta'] != 0])
    diff['num_changed'] = num_grades_changed

    return diff


def _sign(x):
    return -1.0 if x < 0.0 else 1.0


def _biased_random_sample(sample_size, prob_of_relevance=0.9):
    sample = np.random.random_sample(sample_size)
    biased_sample = np.ones(sample_size)
    biased_sample[sample >= prob_of_relevance] = 0
    return biased_sample.astype(int)


@Memoize
def dist_of(dcg_delta, std_dev):
    return NormalDist(mu=dcg_delta, sigma=std_dev)


@Memoize
def _universe_probability(actual_dcg_delta, simulated_dcg_delta, std_dev=1):
    # if abs(actual_dcg_delta - simulated_dcg_delta) > (10 * std_dev):
    #    return 0
    actual_universe_distribution = dist_of(actual_dcg_delta, std_dev)
    simulated_universe_distribution = dist_of(simulated_dcg_delta, std_dev)
    universe_prob = actual_universe_distribution.overlap(simulated_universe_distribution)

    return universe_prob


@Memoize
def corpus():
    corpus = pd.read_csv('data/vmware_ir_content.csv')
    return corpus


def _debug_query(diff, query_id):
    """Examine a single query's diff to see what it says about relevance."""
    query = diff[diff['QueryId'] == query_id]
    query = query.merge(corpus(), right_on='f_name', left_on='DocumentId', how='left')

    print("-------- --------")
    for row in query.sort_values('position_before').to_dict('records'):
        print(row['QueryId'], row['raw_text'][:40].replace("\n", " "), '|',
              row['position_before'], '->', row['position_after'], '|',
              f"{row['weight_delta']:.3f} rels:{row['rels']:.2f} not_rels:{row['not_rels']:.2f}")


def likelihood_not_random(diff, actual_dcg_delta):
    best_case_dcg_delta = diff.iloc[0]['best_case_dcg_delta']

    # Hacky attempt to estimate the probablity the observed DCG is not random
    num_grades_changed = len(diff.loc[diff['weight_delta'] != 0])
    return (num_grades_changed**(actual_dcg_delta / best_case_dcg_delta)) / num_grades_changed


def estimate_relevance(diff, actual_dcg_delta, min_rounds=1000, converge_std_dev=0.02, verbose=False,
                       dcg_diff_std_dev=None):
    """Simulate a single ranking diff and account for the actual dcg delta by guessing result relevance.

    For each oserved query/doc ranking change:
    - 'rels' counts plausible simulations where query/doc grade=1 (relevant),
    - 'not_rels' when its grade=0 (not relevant).

    Essentially each 'rels' / 'not_rels' are counting a bernouli process (coin flip), and represent a binomial distribution
    https://en.wikipedia.org/wiki/Binomial_distribution

    Combining this binomial distribution with other diffs is left as an exercise for the caller. Simply summing can underweight very certain scenarios,
    relative to much more uncertain ones. So you may wish to consider `prob_not_random` and `plausible_universes` to adjust
    rels/not_rels before combining with other diffs.

    Returns modified diff dataframe adding columns:

        - rels: proportion of universes observed to be relevant (grade=1) to account for actual dcg delta
        - not_rels:  proportion of universes observed to be irrelevant (grade=0) to account for actual dcg delta
        - prob_not_random: probability a randomly chosen position that was moved up or down is relevant
          - this is kind of a uniform distribution over the changed relevance, and can be thought of as
            kind of a context-indepentent prior (here context would be the DCG weight of the position changed)
        - plausible_universes: the total probability density of universes that occured in this simulation
          before rels and not_rels converged.

          The more universes that occured, the more simulations were required, and less certain the predictions are.


    """
    num_grades_changed = diff['num_changed'].iloc[0]
    diff['dcg_delta_per_changed'] = actual_dcg_delta / num_grades_changed

    best_case_dcg_delta = diff.iloc[0]['best_case_dcg_delta']

    # Hacky attempt to estimate the probablity the observed DCG is not random
    num_grades_changed = len(diff.loc[diff['weight_delta'] != 0])
    diff['prob_not_random'] = likelihood_not_random(diff, actual_dcg_delta)

    if dcg_diff_std_dev is None:
        dcg_diff_std_dev = 0.01 * np.sqrt(diff['QueryId'].nunique())

    # likelihood_not_random is 0 -> 0.5
    # likelihood_not_random is 1 -> 0.9
    prob_positive = (((diff['prob_not_random'].iloc[0] / 2) + 0.5) - 0.01)
    best_universe_prob = 0.0

    plausible_universe_prob = 0

    # If the probability of observing actual DCG at random is pretty high
    # then we shouldn't increment rels and not_rels too much
    # otherwise rels and not_rels become too certain around
    #
    # OTOH if the probability is very unlikely, the actual DCG delta
    # is very important to account for
    #
    # We can account for this when we normalize rels and not_rels
    diff['grade_changed'] = False
    diff['actual_dcg_delta'] = actual_dcg_delta
    moves_up = diff['weight_delta'] > 0
    num_moves_up = len(diff.loc[moves_up])
    moves_down = diff['weight_delta'] < 0
    num_moves_down = len(diff.loc[moves_down])
    changed = diff['weight_delta'] != 0

    diff.loc[:, 'grade'] = 0
    diff.loc[changed, 'grade_changed'] = True

    start = perf_counter()

    converge_var = converge_std_dev**2
    biggest_var = 1000
    min_universe_prob = 0

    learning_rate = 0.0001
    rounds = 0
    while True:
        # Assign the items with a positive weight delta (moved UP) a relevance of 1
        # with probability `prob_positive` (and conversely for negatives)
        rand_grades_positive = _biased_random_sample(num_moves_up,
                                                     prob_of_relevance=prob_positive)
        rand_grades_negative = _biased_random_sample(num_moves_down,
                                                     prob_of_relevance=1.0 - prob_positive)

        diff.loc[moves_up, 'grade'] = rand_grades_positive
        diff.loc[moves_down, 'grade'] = rand_grades_negative

        # DCG delta of this simulated universe - how close is it to the observed DCG delta?
        simulated_dcg_delta = (diff['grade'] * diff['weight_delta']).sum()
        universe_prob = _universe_probability(actual_dcg_delta, simulated_dcg_delta,
                                              dcg_diff_std_dev)

        # Increment rels and not_rels in proportion to probability of the universe being real
        # how close to observed universe relative to how many possible universes could be THE universe (num_grades_changed)

        if universe_prob > min_universe_prob:
            diff.loc[(diff['grade'] == 1) & changed, 'rels'] += universe_prob
            diff.loc[(diff['grade'] == 0) & changed, 'not_rels'] += universe_prob

            # If rels / not_rels were alpha / beta in beta distribution params
            rels_times_not_rels = diff.loc[changed, 'rels'] * diff.loc[changed, 'not_rels']
            rels_plus_not_rels = diff.loc[changed, 'rels'] + diff.loc[changed, 'not_rels']

            variances = (rels_times_not_rels /
                         ((rels_plus_not_rels**2) * (1 + rels_plus_not_rels)))

            # As binomial distribution (the multiplication of n - number of trials - is
            # already in the accumulating values
            # However, this never converges, and only ever increases with more trials
            # variances = rels_times_not_rels

            biggest_var = variances.max()

        # increment in the direction of the weight delta
        # inversely proportion to the probability of the universe being real
        # (ie we keep the probability of positive prior close to real universes)
        delta = actual_dcg_delta - simulated_dcg_delta
        update_scaled = 1 - universe_prob
        update = learning_rate * update_scaled * _sign(delta)
        prob_positive += update
        plausible_universe_prob += universe_prob

        if universe_prob > best_universe_prob:
            best_universe_prob = universe_prob

        if verbose and rounds % 100 == 0:
            not_rand = diff['prob_not_random'].iloc[0]
            msg = f"Sim: {simulated_dcg_delta:.2f}, Act: {actual_dcg_delta:.2f}({dcg_diff_std_dev:.3f}), Best: {best_case_dcg_delta:.3f} | "
            msg += f"TotUniv: {plausible_universe_prob:.3f} | Var? {biggest_var:.4f}=>{converge_var:.4f}"
            msg += f"| Upd {update:.4f}, Draw {prob_positive:.4f}, NotRand: {not_rand:.5f}"
            msg += f" | {num_grades_changed} changed | {rounds}/{perf_counter() - start:.3f}s"
            print(msg)
        rounds += 1

        if biggest_var <= converge_var and rounds >= min_rounds:
            diff['prob_positive'] = prob_positive
            break

    # rels - proportion of plausible universes that have this grade as 1
    # not_rels - proportion of plausible universes that have this grade as 0
    # plausible_universes - accumulated probability density of all explored universes
    diff['rels'] /= plausible_universe_prob
    diff['not_rels'] /= plausible_universe_prob
    diff['plausible_universes'] = plausible_universe_prob
    diff['best_universe_prob'] = best_universe_prob
    prob_not_random = diff['prob_not_random'] = (prob_positive - 0.5) * 2

    print(f"BestCase: {best_case_dcg_delta}; Observed: {actual_dcg_delta}")
    print(f"Likelihood: {prob_not_random:.2f} | Total Universe Prob: {plausible_universe_prob:.2f}")
    cols = ['QueryId', 'DocumentId', 'position_before', 'position_after', 'weight_delta', 'rels', 'not_rels']
    print(diff[diff['grade_changed']][cols].sort_values('rels', ascending=False))

    # _debug_query(diff, 0)
    # _debug_query(diff, 1)
    return diff
