from numpy import dot
from numpy.linalg import norm
from vmware.vector.use import encode as encode_use
from vmware.vector.mpnet import encode as encode_mpnet


encoders = {
    'use': encode_use,
    'mpnet': encode_mpnet
}


def use_rescore_query(es_body, rescore_depth, query, vector_field='raw_text_use'):
    query_vector = encode_use(query)
    es_body['rescore'] = {
        "window_size": rescore_depth,
        "query": {
            "rescore_query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
    }
    return es_body


encoded_query_use_cache = {}


def passage_similarity_long_lines(query, hit,
                                  verbose=False,
                                  remaining_lines=True):
    # start = perf_counter()
    source = hit['_source']
    lines = [source['first_line']]
    for line in source['remaining_lines']:
        if len(line) > 20:
            lines.append(line)

    query_encodings = {model_name: model(query) for model_name, model in encoders.items()}
    max_sim = {model_name: -1.0 for model_name in encoders}
    sum_sim = {model_name: 0.0 for model_name in encoders}

    num_lines = 0
    if verbose:
        print('----')
        print(query)
    for line in lines:
        cos_sims = {}
        sum_sims = {}
        max_sims = {}
        for model_name, model in encoders.items():
            encoding = model(line)
            cos_sims[model_name] = (dot(encoding, query_encodings[model_name]) / (norm(encoding) * norm(query_encodings[model_name])))
            sum_sims[model_name] = sum_sim[model_name] + cos_sims[model_name]
            max_sims[model_name] = max(max_sim[model_name], cos_sims[model_name])
        first_key = list(encoders.keys())[0]
        num_stars = 10 * (cos_sims[first_key] + 1)
        num_lines += 1
        if verbose:
            print(f"{cos_sims[first_key]:.2f}", "*" * int(num_stars), " " * (20 - int(num_stars)), line[:40])
        if not remaining_lines:
            break

    for model_name, model in encoders.items():
        mean = sum_sims[model_name] / num_lines
        hit['_source'][f'max_sim_{model_name}'] = max_sims[model_name]
        hit['_source'][f'sum_sim_{model_name}'] = sum_sims[model_name]
        hit['_source'][f'mean_sim_{model_name}'] = mean

    if verbose:
        for model_name, model in encoders.items():
            print(f"{model_name} MAX: {max_sims[model_name]:.2f} | SUM: {sum_sims[model_name]:.2f} | MEAN: {mean:.2f}")
