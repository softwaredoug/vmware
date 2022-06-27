import pandas as pd
import math
import pytest
from vmware.simulate.diff_simulation import create_results_diff, estimate_relevance, best_runs


def dcg_weight_at(rank):
    return 1 / (math.log(rank + 1, 2))


@pytest.fixture
def single_result():
    results = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "1234"},  # Not relevant
        {'QueryId': 1, 'DocumentId': "5678"}   # Relevant (grade=1)
    ])
    dcg = dcg_weight_at(2)
    return results, dcg


@pytest.fixture
def best_case_diff():
    results_before = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "1234"},  # relevance = 0 * 1
        {'QueryId': 1, 'DocumentId': "5678"}   # relevance = 1 *
    ])

    results_after = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "5678"},
        {'QueryId': 1, 'DocumentId': "1234"}
    ])

    diff = create_results_diff(results_before, results_after)
    actual_dcg_delta = dcg_weight_at(1) - dcg_weight_at(2)
    return diff, actual_dcg_delta


@pytest.fixture
def ambiguous_diff():
    results_before = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "1234"},
        {'QueryId': 1, 'DocumentId': "5678"},
        {'QueryId': 2, 'DocumentId': "1234"},
        {'QueryId': 2, 'DocumentId': "5678"}
    ])

    both_queries_swapped = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "5678"},
        {'QueryId': 1, 'DocumentId': "1234"},
        {'QueryId': 2, 'DocumentId': "5678"},
        {'QueryId': 2, 'DocumentId': "1234"}
    ])

    # Plausible scenarios explain this DCG change
    # Three relevant docs:
    # doc 1234 and doc 5678 are both relevant for query 1, but only 5678 is relevant for query 2
    # doc 1234 and doc 5678 are both relevant for query 2, but only 5678 is relevant for query 1
    # One relevant doc:
    # doc 1234 and doc 5678 are NOT relevant for query 1, but only 5678 is relevant for query 2
    # doc 1234 and doc 5678 are NOT relevant for query 2, but only 5678 is relevant for query 1
    #
    # To capture all of these, many simulations (total_universe_prob) need to be run to see this
    #
    diff = create_results_diff(results_before, both_queries_swapped)
    actual_dcg_delta = dcg_weight_at(1) - dcg_weight_at(2)
    return diff, actual_dcg_delta


def test_best_case_prob_not_random_not_random(best_case_diff):
    diff, actual_dcg_delta = best_case_diff
    diff = estimate_relevance(diff, actual_dcg_delta)

    judgment_1234 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "1234")]
    judgment_5678 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "5678")]

    assert judgment_1234['prob_not_random'].iloc[0] == pytest.approx(1.0, 0.05), \
        "Given the diff is best case, prob_not_random should be high"
    assert judgment_5678['prob_not_random'].iloc[0] == pytest.approx(1.0, 0.05), \
        "Given the diff is best case, prob_not_random should be high"


def test_best_case_entropy_less_than_ambiguous_case(best_case_diff, ambiguous_diff):
    best_case_diff, best_case_actual_dcg_delta = best_case_diff
    best_case_diff = estimate_relevance(best_case_diff, best_case_actual_dcg_delta)

    ambig_case_diff, ambig_case_actual_dcg_delta = ambiguous_diff
    ambig_case_diff = estimate_relevance(ambig_case_diff, ambig_case_actual_dcg_delta)

    entropies = best_case_diff.merge(ambig_case_diff, how='left', on=['QueryId', 'DocumentId'])[['entropy_x', 'entropy_y']]
    assert (entropies['entropy_x'] < entropies['entropy_y']).all()


def test_best_case_trumps_ambig_case(best_case_diff, ambiguous_diff):
    best_case_diff, best_case_actual_dcg_delta = best_case_diff
    best_case_diff = estimate_relevance(best_case_diff, best_case_actual_dcg_delta)

    ambig_case_diff, ambig_case_actual_dcg_delta = ambiguous_diff
    ambig_case_diff = estimate_relevance(ambig_case_diff, ambig_case_actual_dcg_delta)

    all_diffs = pd.concat([best_case_diff, ambig_case_diff])
    combined = best_runs(all_diffs)

    # Best case has unambiguous information that should be taken heavily
    poor_doc_rels = combined.loc[(1, '1234')]['rels']
    poor_doc_not_rels = combined.loc[(1, '1234')]['not_rels']
    assert (10 * poor_doc_rels) < poor_doc_not_rels

    good_doc_rels = combined.loc[(1, '5678')]['rels']
    good_doc_not_rels = combined.loc[(1, '5678')]['not_rels']
    assert (10 * good_doc_not_rels) < good_doc_rels


def test_best_case_diff_certain(best_case_diff):
    diff, actual_dcg_delta = best_case_diff
    diff = estimate_relevance(diff, actual_dcg_delta)

    judgment_1234 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "1234")]
    judgment_5678 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "5678")]

    assert judgment_5678['rels'].iloc[0] > judgment_1234['rels'].iloc[0]
    assert judgment_5678['not_rels'].iloc[0] < judgment_1234['not_rels'].iloc[0]
    # They should be VERY different
    assert judgment_5678['rels'].iloc[0] - judgment_1234['rels'].iloc[0] > 0.9, \
        "Given the diff is best case, rels diff should be very high"
    assert judgment_1234['not_rels'].iloc[0] - judgment_5678['not_rels'].iloc[0] > 0.9, \
        "Given the diff is best case, not_rels diff should be very high"


def test_either_swap_accounts_for_dcg_diff(ambiguous_diff):
    diff, actual_dcg_delta = ambiguous_diff
    diff = estimate_relevance(diff,
                              actual_dcg_delta,
                              verbose=True)

    judgment_1_1234 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "1234")]
    judgment_1_5678 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "5678")]
    judgment_2_1234 = diff[(diff['QueryId'] == 2) & (diff['DocumentId'] == "1234")]
    judgment_2_5678 = diff[(diff['QueryId'] == 2) & (diff['DocumentId'] == "5678")]

    assert judgment_1_5678['rels'].iloc[0] > judgment_1_5678['not_rels'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"
    assert judgment_1_1234['not_rels'].iloc[0] > judgment_1_1234['rels'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"
    assert judgment_2_5678['rels'].iloc[0] > judgment_2_5678['not_rels'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"
    assert judgment_2_1234['not_rels'].iloc[0] > judgment_2_1234['rels'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"

    assert judgment_1_5678['rels'].iloc[0] == pytest.approx(judgment_2_5678['rels'].iloc[0], 0.1), \
        "Both 5678 swaps could account for DCG delta, so should be approximately equal"
    assert judgment_1_5678['not_rels'].iloc[0] == pytest.approx(judgment_2_5678['not_rels'].iloc[0], 0.1), \
        "Both 5678 swaps could account for DCG delta, so should be approximately equal"


def test_ambiguous_diff_is_less_certain(ambiguous_diff):
    diff, actual_dcg_delta = ambiguous_diff
    diff = estimate_relevance(diff,
                              actual_dcg_delta,
                              verbose=True)

    judgment_1_1234 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "1234")]
    judgment_1_5678 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "5678")]
    judgment_2_1234 = diff[(diff['QueryId'] == 2) & (diff['DocumentId'] == "1234")]
    judgment_2_5678 = diff[(diff['QueryId'] == 2) & (diff['DocumentId'] == "5678")]

    assert judgment_1_5678['rels'].iloc[0] < 0.8, \
        "While most simulations should have this as more relevant, it should have many where it is not the one swapped"
    assert judgment_2_5678['rels'].iloc[0] < 0.8, \
        "While most simulations should have this as more relevant, it should have many where it is not the one swapped"

    assert judgment_1_1234['not_rels'].iloc[0] > 0.2, \
        "While most simulations should have this as more irrelevant, it should have many where it is not the one swapped"
    assert judgment_2_1234['not_rels'].iloc[0] > 0.2, \
        "While most simulations should have this as more irrelevant, it should have many where it is not the one swapped"


def test_can_create_fake_diff_from_result_in_isolation(single_result):
    results, dcg = single_result
    diff = create_results_diff(results_before=None, results_after=results)
    assert diff['weight_delta'].iloc[0] == pytest.approx(1.0)
    assert diff['weight_delta'].iloc[1] == pytest.approx(0.63093)


def test_can_reconstruct_label_from_result_in_isolation(single_result):
    results, dcg = single_result
    diff = create_results_diff(results_before=None, results_after=results)
    diff = estimate_relevance(diff,
                              dcg,
                              verbose=True)
    diff[diff['DocumentId'] == 5678]['rels'] = pytest.approx(1.0, 0.01)
    diff[diff['DocumentId'] == 5678]['not_rels'] = pytest.approx(0.0, 0.01)
    diff[diff['DocumentId'] == 1234]['rels'] = pytest.approx(0.0, 0.01)
    diff[diff['DocumentId'] == 1234]['not_rels'] = pytest.approx(1.0, 0.01)
