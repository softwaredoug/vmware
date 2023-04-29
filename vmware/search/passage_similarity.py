from numpy import dot
import redis
from time import perf_counter
from numpy.linalg import norm
from vmware.vector.use import encode as encode_use
from vmware.vector.mpnet import encode as encode_mpnet


encoders = {
    'use': encode_use,
    'mpnet': encode_mpnet
}


# doc_sims_cache = {}
r = redis.Redis(host='localhost', port=6379)


def cached_fields():
    fields = []
    for model_name, model in encoders.items():
        fields.extend([f"first_line_sim_{model_name}",
                       f"max_sim_5_{model_name}",
                       f"max_sim_{model_name}",
                       f"max_sim_10_{model_name}",
                       f"sum_sim_{model_name}",
                       f"sum_sim_5_{model_name}",
                       f"sum_sim_10_{model_name}",
                       f"mean_sim_{model_name}",
                       f"mean_sim_5_{model_name}",
                       f"mean_sim_10_{model_name}"])
    return fields


def sims_from_cache(cache_key, hit):
    fields = cached_fields()

    sims = r.hmget(cache_key, fields)
    if sims is None:
        return False

    for field, sim in zip(fields, sims):
        if sim is None:
            return False
        hit['_source'][field] = float(sim)
    return True


def sims_to_cache(cache_key, hit):
    fields = cached_fields()

    for field in fields:
        r.hset(cache_key, field, float(hit['_source'][field]))


def passage_similarity_long_lines(query, hit,
                                  verbose=False,
                                  remaining_lines=True,
                                  sim_cache=True,
                                  vector_cache=False):
    start_time = perf_counter()
    encode_time = 0.0
    cache_key = f"{query}|||{hit['_id']}"
    if sim_cache and sims_from_cache(cache_key, hit):
        for field in cached_fields():
            assert field in hit['_source']
        if verbose:
            print(f"Cached at {cache_key}")
        return

    source = hit['_source']
    lines = [source['first_line']]
    for line in source['remaining_lines']:
        if len(line) > 20:
            lines.append(line)

    query_encodings = {model_name: model(query) for model_name, model in encoders.items()}

    max_sims = {model_name: -1.0 for model_name in encoders}
    max_sims5 = {model_name: -1.0 for model_name in encoders}
    max_sims10 = {model_name: -1.0 for model_name in encoders}
    sum_sims = {model_name: 0.0 for model_name in encoders}
    sum_sims10 = {model_name: 0.0 for model_name in encoders}
    sum_sims5 = {model_name: 0.0 for model_name in encoders}
    first_line_sim = {model_name: 0.0 for model_name in encoders}

    num_lines = 0
    if verbose:
        print('----')
        print(query)
    for idx, line in enumerate(lines):
        cos_sims = {}
        for model_name, model in encoders.items():
            encode_start = perf_counter()
            encoding = model(line, cached=vector_cache)
            encode_time += (perf_counter() - encode_start)
            cos_sims[model_name] = (dot(encoding, query_encodings[model_name]) / (norm(encoding) * norm(query_encodings[model_name])))
            sum_sims[model_name] = sum_sims[model_name] + cos_sims[model_name]
            max_sims[model_name] = max(max_sims[model_name], cos_sims[model_name])

            if idx < 10:
                sum_sims10[model_name] = sum_sims10[model_name] + cos_sims[model_name]
                max_sims10[model_name] = max(max_sims10[model_name], cos_sims[model_name])
            if idx < 5:
                sum_sims5[model_name] = sum_sims5[model_name] + cos_sims[model_name]
                max_sims5[model_name] = max(max_sims5[model_name], cos_sims[model_name])
            if idx == 0:
                first_line_sim[model_name] = cos_sims[model_name]

        first_key = list(encoders.keys())[0]
        num_stars = 10 * (cos_sims[first_key] + 1)
        num_lines += 1
        if verbose:
            print(f"{first_key} {cos_sims[first_key]:.2f}", "*" * int(num_stars), " " * (20 - int(num_stars)), line[:40])
        if not remaining_lines:
            break

    for model_name, model in encoders.items():
        mean = sum_sims[model_name] / num_lines
        mean10 = sum_sims10[model_name] / min(num_lines, 10)
        mean5 = sum_sims5[model_name] / min(num_lines, 5)
        assert mean10 <= 1.0
        assert mean5 <= 1.0
        assert mean <= 1.0
        # assert mean != sum_sims[model_name]
        hit['_source'][f'max_sim_{model_name}'] = max_sims[model_name]
        hit['_source'][f'sum_sim_{model_name}'] = sum_sims[model_name]
        hit['_source'][f'mean_sim_{model_name}'] = sum_sims[model_name]

        hit['_source'][f'max_sim_10_{model_name}'] = max_sims10[model_name]
        hit['_source'][f'sum_sim_10_{model_name}'] = sum_sims10[model_name]
        hit['_source'][f'mean_sim_10_{model_name}'] = mean10

        hit['_source'][f'max_sim_5_{model_name}'] = max_sims5[model_name]
        hit['_source'][f'sum_sim_5_{model_name}'] = sum_sims5[model_name]
        hit['_source'][f'mean_sim_5_{model_name}'] = mean5

        hit['_source'][f'first_line_sim_{model_name}'] = first_line_sim[model_name]

    sims_to_cache(cache_key, hit)
    for field in cached_fields():
        assert field in hit['_source']

    if verbose:
        for model_name, model in encoders.items():
            print(f"{model_name} MAX: {max_sims[model_name]:.2f} | SUM: {sum_sims[model_name]:.2f} | MEAN: {mean:.2f}")
            print(f"{model_name} MAX10: {max_sims10[model_name]:.2f} | SUM10: {sum_sims10[model_name]:.2f} | MEAN10: {mean10:.2f}")
        print(f"Took: {perf_counter() - start_time:.2f} | Enc {encode_time:.2f}")
        print(f"Cache: sim:{sim_cache} vector_cache:{vector_cache}")
