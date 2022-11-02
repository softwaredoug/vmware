import numpy as np
import pandas as pd


def exact_nearest_neighbors(query_vector, matrix, n=100):
    """ nth nearest neighbors as array
        with indices of nearest neighbors"""
    # exact_nearest_neighbors.normed = np.linalg.norm(matrix, axis=1)

    dotted = np.dot(matrix, query_vector)
    # nn = np.divide(dotted, exact_nearest_neighbors.normed)
    top_n = np.argpartition(-nn, n)[:n]
    return top_n, nn[top_n]


def similarity(query, encoder, corpus, column, n=10):
    query_vector = encoder(query)
    vectors = corpus[column]
    vectors = np.array(vectors.to_list())

    top_n, scores = exact_nearest_neighbors(query_vector, vectors, n=n)
    top_n = pd.DataFrame({'icol': top_n, 'scores': scores})
    top_n = top_n.set_index('icol').sort_values('scores', ascending=False)
    top_n_corpus = corpus.iloc[top_n.index].copy()
    top_n_corpus['scores'] = sorted(scores, reverse=True)

    return top_n_corpus
