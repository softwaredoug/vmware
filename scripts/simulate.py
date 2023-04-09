# Can we infer relevance grades from just a difference in the mean NDCG of two samples?
import path  # noqa
import pandas as pd
from vmware.simulate.multiple import run_diffs, significant_diffs, grade_judgments, debugable_results


# Submissions from the kaggle vmware competition
# NDCG at 5
ndcgs = {
    None: 0.0,  # Causes each NDCG to be looked at in isolation, not just as a diff
    # 'runs/turnbull_submission_1652543571065569.csv': 0.0,   # This one was submitted on accident, but still valid info
    'runs/noise.csv': 0.00060,          # Randomly shuffled results, everything here should end up irrelevant for its query
    'runs/use_feedback_rrf_turnbull_submission_1653226391886872.csv': 0.16806,
    'runs/chatgpt_mlt_turnbull_168095807535608.csv': 0.20775,
    'runs/turnbull_submission_1652544680901428.csv': 0.20911,
    'runs/simulated_labels_turnbull.csv': 0.26169,
    'runs/simulated_every_combination_unweighted_turnbull_16561689617190552.csv': 0.28237,
    'runs/simulated_labels_turnbull_2.csv': 0.28722,
    'runs/best_compound_form_turnbull_1654044184531847.csv': 0.28790,
    'runs/max_passage_rerank_first_remaining_lines_turnbull_1653950500881539.csv': 0.29467,
    'runs/pull_out_firstline_turnbull_1653253060074837.csv': 0.29668,
    'runs/sum_use_at_50_rerank_turnbull_16538316179778419.csv': 0.29762,
    'runs/bm25_raw_text_to_remaining_lines_search_turnbull_16538323991927738.csv': 0.30396,
    'runs/sum_use_at_5_rerank_turnbull_16538284889842649.csv': 0.30550,
    'runs/max_use_at_5_rerank_turnbull_16538277759430668.csv': 0.30562,
    'runs/with_best_compounds_at_5_only_phrase_search_turnbull_16544332515351.csv': 0.30642,
    'runs/rerank_slop_search_remaining_lines_max_snippet_at_5_turnbull_1654439885030507.csv': 0.31574,
    'runs/with_best_compounds_at_5_only_phrase_search_turnbull_1654439995765457.csv': 0.31643,
    'runs/max_passage_rerank_at_5_attempt_2_turnbull_1653833337447114.csv': 0.31569,
    'runs/with_best_compounds_at_50_plus_10_times_use_turnbull_165445182567455.csv': 0.32681,
}


def read_csv(fname):
    if fname is None:
        return None
    else:
        return pd.read_csv(fname)


def main():
    judgments = pd.DataFrame(columns=['QueryId', 'DocumentId', 'alpha', 'beta',
                                      'weight_delta', 'position_delta'])
    runs = [(run[0], read_csv(run[0]), run[1]) for run in ndcgs.items()]

    runs = significant_diffs(runs)
    judgments = run_diffs(runs)
    judgments = grade_judgments(judgments)

    # Join with corpus for debugging
    judgments = judgments.reset_index()
    results = debugable_results(judgments)
    results.to_csv('data/simulated_results.csv', index=False)


if __name__ == "__main__":
    main()
