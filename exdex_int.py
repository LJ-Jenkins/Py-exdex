import math
import numpy as np
from itertools import groupby

def gdd_theta(theta, q_u, D):
    d = q_u * D
    etd = math.exp(theta * d)
    val = -(theta * d**2 * etd - 2 * d * etd + 1) / (etd - theta)**2
    return val

def split_by_NAs(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    temp = np.apply_along_axis(lambda col: np.array(np.split(col, np.where(np.diff(np.isnan(col)))[0] + 1)), axis=0, arr=x)
    column = np.repeat(np.arange(x.shape[1]), np.vectorize(lambda t: np.sum(t['values']))(temp))
    if len(temp) == 1:
        max_leng = np.max(temp[0]['lengths'])
        n_seq = np.sum(temp[0]['values'])
    else:
        max_leng = np.max(np.array([t['lengths'] for t in temp]), axis=0)
        n_seq = np.sum(np.array([t['values'] for t in temp]), axis=0)
    from_fn = lambda t: np.cumsum(t['lengths']) - t['lengths'] + 1 if len(t) > 0 else []
    to_fn = lambda t: np.cumsum(t['lengths']) if len(t) > 0 else []
    from_rows = np.concatenate([from_fn(t) for t in temp])
    to_rows = np.concatenate([to_fn(t) for t in temp])
    newx_fn = lambda i, from_r, to_r, col: np.concatenate([x[from_r[i]:to_r[i], col[i]], np.repeat(np.nan, max_leng - (to_r[i] - from_r[i] + 1))])
    newx = np.array([newx_fn(i, from_rows, to_rows, column) for i in range(n_seq)])
    return newx

def split_by_nans(data):
    non_nan_groups = []
    for key, group in groupby(data, lambda x: math.isnan(x)):
        if not key:
            non_nan_groups.append(list(group))
    groups = non_nan_groups
    max_group_size = max(len(group) for group in groups)
    padded_groups = []
    for group in groups:
        padding_size = max_group_size - len(group)
        padded_group = group + [float('nan')] * padding_size
        padded_group = np.array(np.array(padded_group, dtype="object"), dtype="float")
        padded_group = np.reshape(padded_group, (padded_group.size,1))
        padded_groups.append(padded_group)
    new_data = np.hstack(padded_groups)
    return new_data

# fini