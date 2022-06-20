# Can we infer relevance grades from just a difference in the mean NDCG of two samples?
import path  # noqa
import pandas as pd
from vmware.simulate.multiple import run_diffs, significant_diffs, grade_judgments


# Submissions from the kaggle vmware competition
# NDCG at 5
ndcgs = {
    'data/noise.csv': 0.00060,
    'data/use_feedback_rrf_turnbull_submission_1653226391886872.csv': 0.16806,
    'data/turnbull_submission_1652544680901428.csv': 0.20911,
    'data/best_compound_form_turnbull_1654044184531847.csv': 0.28790,
    'data/max_passage_rerank_first_remaining_lines_turnbull_1653950500881539.csv': 0.29467,
    'data/pull_out_firstline_turnbull_1653253060074837.csv': 0.29668,
    'data/sum_use_at_50_rerank_turnbull_16538316179778419.csv': 0.29762,
    'data/bm25_raw_text_to_remaining_lines_search_turnbull_16538323991927738.csv': 0.30396,
    'data/with_best_compounds_at_5_only_phrase_search_turnbull_16544332515351.csv': 0.30642,
    'data/rerank_slop_search_remaining_lines_max_snippet_at_5_turnbull_1654439885030507.csv': 0.31574,
    'data/with_best_compounds_at_5_only_phrase_search_turnbull_1654439995765457.csv': 0.31643,
    'data/with_best_compounds_at_50_plus_10_times_use_turnbull_165445182567455.csv': 0.32681,

    # Random noise we assume has NDCG=0
    # 'noise.csv': 0.0
}


def main():
    judgments = pd.DataFrame(columns=['QueryId', 'DocumentId', 'alpha', 'beta',
                                      'weight_delta', 'position_delta'])
    runs = [(run[0], pd.read_csv(run[0]), run[1]) for run in ndcgs.items()]

    runs = significant_diffs(runs)
    judgments = run_diffs(runs)
    judgments = grade_judgments(judgments)

    # Join with corpus for debugging
    corpus = pd.read_csv('data/vmware_ir_content.csv')
    queries = pd.read_csv('data/test.csv')
    judgments = judgments.reset_index()
    results = judgments.merge(corpus, right_on='f_name', left_on='DocumentId', how='left')
    results = results.merge(queries, on='QueryId', how='left')
    results.to_csv('data/simulated_results.csv', index=False)


if __name__ == "__main__":
    main()
