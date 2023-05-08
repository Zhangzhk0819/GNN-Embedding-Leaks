import numpy as np


def concatenate(embedding_1, embedding_2):
    return np.concatenate((embedding_1, embedding_2), axis=1)


def cosine_similarity(embedding_1, embedding_2):
    dot_product = np.sum(embedding_1 * embedding_2, axis=1)
    graph_norm = np.sqrt(np.sum(embedding_1 ** 2, axis=1)) / embedding_1.shape[1]
    subgraph_norm = np.sqrt(np.sum(embedding_2 ** 2, axis=1)) / embedding_2.shape[1]
    cosine_similarity = dot_product / (graph_norm * subgraph_norm)
    return cosine_similarity.reshape([-1, 1])


def l1_distance(embedding_1, embedding_2):
    l1_distance = np.mean(np.abs(embedding_1 - embedding_2), axis=1)
    return l1_distance.reshape([-1, 1])


def l2_distance(embedding_1, embedding_2):
    l2_distance = np.sqrt(np.sum((embedding_1 - embedding_2) ** 2, axis=1)) / embedding_1.shape[1]
    return l2_distance.reshape([-1, 1])


def element_l1(embedding_1, embedding_2):
    return np.abs(embedding_1 - embedding_2)


def element_l2(embedding_1, embedding_2):
    return np.sqrt((embedding_1 - embedding_2) ** 2)
