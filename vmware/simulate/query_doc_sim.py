"""Get query-doc sim scores via passage_similarity.py."""
import pandas as pd
import numpy as np
from time import perf_counter
from elasticsearch import Elasticsearch
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt


import sys
import os
# Prepend cwd to sys.path
sys.path.insert(0, os.getcwd())
from vmware.search.passage_similarity import passage_similarity_long_lines, cached_fields  # noqa: E402


def passage_similarity(es, query, doc_id):
    """Get passage similarity scores."""
    result = es.get(index="vmware", id=doc_id)
    passage_similarity_long_lines(query, result)
    similarities = {}
    for field in cached_fields():
        similarities[field] = float(result['_source'][field])
    return similarities


def sim_worker(es, result):
    es = Elasticsearch()
    query = result['Query']
    doc_id = result['DocumentId']
    start = perf_counter()
    sims = passage_similarity(es, query, doc_id)
    print(query, doc_id, perf_counter() - start)
    return {**result, **sims}


def score_similarity(simulated_results: pd.DataFrame, num_to_score: int = None) -> pd.DataFrame:
    """Score similarity."""
    new_df = []
    es = Elasticsearch()
    tasks = []
    batch_size = min(100, num_to_score) if num_to_score is not None else 100
    with ThreadPoolExecutor(max_workers=1) as executor:
        for idx, result in enumerate(simulated_results.sample(frac=1).to_dict(orient='records')):
            tasks.append(executor.submit(sim_worker, es, result))

            if len(tasks) >= batch_size:
                print(f"Waiting for completion... {idx}, {num_to_score}")
                for task in as_completed(tasks):
                    new_df.append(task.result())
                tasks = []
                if num_to_score is not None and len(new_df) >= num_to_score:
                    break
    return pd.DataFrame(new_df)


def assign_features(scored_results: pd.DataFrame) -> pd.DataFrame:
    for field in cached_fields():
        scored_results[f'{field}_1'] = scored_results[field] * scored_results['grade']
        scored_results[f'{field}_0'] = scored_results[field] * (1.0 - scored_results['grade'])


def build_hist(scored_results, field):
    """Build a bernouli histogram around grade."""
    buckets = np.arange(0, 1.01, 0.01)
    assigned_buckets = np.digitize(scored_results[field], buckets)
    hist1 = np.zeros(100)
    hist0 = np.zeros(100)
    for idx, bucket in enumerate(assigned_buckets):
        hist1[bucket] += scored_results['grade'].iloc[idx]
        hist0[bucket] += (1 - scored_results['grade'].iloc[idx])
    return hist1, hist0


def plot(hist0, hist1):
    x = np.arange(0, 1.00, 0.01)
    plt.axis([0, 1, 0, 1 + max(max(hist0), max(hist1))])
    weighted_sum0 = np.sum(np.array(hist0) * np.array(np.arange(0, 1.00, 0.01)))
    mean0 = weighted_sum0 / np.sum(hist0)
    weighted_sum1 = np.sum(np.array(hist1) * np.array(np.arange(0, 1.00, 0.01)))
    mean1 = weighted_sum1 / np.sum(hist1)
    y0 = hist0
    y1 = hist1
    # Plot 0s as red
    plt.bar(x, y0, color='r', width=0.01)
    # Plot 1s as green
    plt.bar(x, y1, color='g', width=0.01)

    plt.axvline(x=mean0, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(x=mean1, color='g', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/simulated_results.csv")
    new_df = score_similarity(df)
