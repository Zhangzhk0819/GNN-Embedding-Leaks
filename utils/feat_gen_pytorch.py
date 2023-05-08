import torch


def concatenate(embedding_1, embedding_2):
    return torch.cat((embedding_1, embedding_2), dim=1)


def cosine_similarity(embedding_1, embedding_2):
    dot_product = torch.sum(embedding_1 * embedding_2, dim=1)
    graph_norm = torch.sqrt(torch.sum(embedding_1 ** 2, dim=1)) / embedding_1.shape[1]
    subgraph_norm = torch.sqrt(torch.sum(embedding_2 ** 2, dim=1)) / embedding_2.shape[1]
    cosine_similarity = dot_product / (graph_norm * subgraph_norm)
    cosine_similarity = cosine_similarity.float()
    return cosine_similarity.reshape([-1, 1])


def l1_distance(embedding_1, embedding_2):
    l1_distance = torch.mean(torch.abs(embedding_1 - embedding_2), dim=1).float()
    return l1_distance.reshape([-1, 1])


def l2_distance(embedding_1, embedding_2):
    l2_distance = torch.sqrt(torch.sum((embedding_1 - embedding_2) ** 2, dim=1)) / embedding_1.shape[1]
    l2_distance = l2_distance.float()
    return l2_distance.reshape([-1, 1])


def element_l1(embedding_1, embedding_2):
    return torch.abs(embedding_1 - embedding_2).float()


def element_l2(embedding_1, embedding_2):
    return torch.sqrt((embedding_1 - embedding_2) ** 2).float()
