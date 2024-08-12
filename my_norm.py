from collections import defaultdict
import numpy as np
import torch
from my_sinkhorn import sinkhorn_raw

def norm_by_prefix(collection, prefix):
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    from copy import deepcopy
    new_collection = deepcopy(collection)
    global_mean = collection.mean(axis=0)
    global_var = collection.var(axis=0)
    for p, p_code in prefix_code.items():
        p_collection = collection[p_code]
        mean_value = p_collection.mean(axis=0)
        var_value = p_collection.var(axis=0)
        var_value[var_value == 0] = 1
        scale = global_var / var_value
        scale[np.isnan(scale)] = 1
        scale = 1
        p_collection = (p_collection - mean_value + global_mean) * scale
        new_collection[p_code] = p_collection
    return new_collection


def center_pq(m, prefix):
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    from copy import deepcopy
    new_m = deepcopy(m)
    for p, p_code in prefix_code.items():
        sub_m = m[p_code]
        new_m[p_code] = sub_m.mean(axis=0)
    return new_m


def norm_code_by_prefix(collection, centroids, prefix, epsilon=1):
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    attention = np.matmul(collection, centroids.T)
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    code = [None for _ in range(len(collection))]
    for p, p_code in prefix_code.items():
        p_collection = attention[p_code]
        distances = p_collection
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered_distances = (distances - middle) / amplitude
        distances = torch.tensor(centered_distances)
        Q = sinkhorn_raw(
            distances,
            epsilon,
            100,
            use_distrib_train=False
        )  # B-K
        codes = torch.argmax(Q, dim=-1).tolist()
        for i, c in zip(p_code, codes):
            code[i] = c
    return code