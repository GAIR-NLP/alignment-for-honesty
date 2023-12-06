import random


def random_sample(l, sample_size, seed=42):
    if sample_size >= len(l):
        return l
    random.seed(seed)
    possible_idxs = list(range(len(l)))
    sample_idxs = random.sample(possible_idxs, sample_size)
    return [l[idx] for idx in sample_idxs]
