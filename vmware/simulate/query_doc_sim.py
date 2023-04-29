"""Get query-doc sim scores via passage_similarity.py."""
import pandas as pd
from time import perf_counter
from elasticsearch import Elasticsearch
from vmware.search.passage_similarity import passage_similarity_long_lines, cached_fields
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def score_similarity(simulated_results: pd.DataFrame):
    """Score similarity."""
    new_df = []
    es = Elasticsearch()
    tasks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, result in enumerate(simulated_results.sample(frac=1).to_dict(orient='records')):
            tasks.append(executor.submit(sim_worker, es, result))

            if len(tasks) >= 100:
                for task in as_completed(tasks):
                    new_df.append(task.result())
                tasks = []
    return pd.DataFrame(new_df)
