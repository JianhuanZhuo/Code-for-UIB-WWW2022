import os
import pickle
from collections import defaultdict
import zipfile

import torch
from tqdm import tqdm


def cache_or(cache_name, folder=None, *, generator: callable, abort_if_not_exist=False):
    if not cache_name.endswith(".pkl"):
        raise NotImplemented
    if not cache_name.startswith("cache."):
        raise NotImplemented

    abs_name = os.path.abspath(os.path.join(folder, cache_name)) if folder else os.path.abspath(cache_name)

    zip_file = abs_name + ".zip"
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, mode='r') as zf:
            assert len(zf.namelist()) == 1 and cache_name in zf.namelist()
            with zf.open(zf.namelist()[0], mode='r') as f:
                return pickle.load(f)

    if os.path.exists(abs_name):
        if not abort_if_not_exist:
            print(f"reading {cache_name}")
        with open(abs_name, 'rb') as f:
            res = pickle.load(f)
            if not abort_if_not_exist:
                print(f"   read {cache_name}")
            return res
    elif abort_if_not_exist:
        raise FileNotFoundError(f"cache not found : {abs_name}")
    else:
        print(f"generating data: {cache_name}")
        result = generator()
        print(f"  pickling data: {cache_name}")
        with open(abs_name, 'wb') as f:
            pickle.dump(result, f)
        print(f"    processing over: {cache_name}")
        return result


def group_kv(kvs, tqdm_title=None):
    result = defaultdict(set)
    if tqdm_title:
        kvs = tqdm(kvs, desc=tqdm_title)
    for k, v in kvs:
        result[k].add(v)
    return dict(result)


def clip_by_norm(value, clip):
    """
    参考：https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
    """
    assert len(value.shape) == 2
    assert type(clip) == float or type(clip) == int
    num, dim = value.shape
    norms = value.norm(p=2, dim=1, keepdim=True)
    assert norms.shape == torch.Size([num, 1])
    clip_mask = (norms > clip).float()
    res = (1 - clip_mask) * value + clip_mask * (value * clip / norms)
    assert res.shape == torch.Size([num, dim])
    return res
