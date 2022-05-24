import numpy as np
import torch
from numpy import random
from tqdm.auto import tqdm

from tools.utils import cache_or


class SamplingSpace:
    """
    快速采样方法
    """

    def __init__(self, num_user, num_item, dataset, device,
                 enable_sampling_record=True,
                 enable_sampling_negative=True,
                 enable_sampling_positive=True,
                 train_set=None,
                 title=None,
                 cache_base=None,
                 cache_folder=None,
                 in_grid=False,
                 ):
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.records_len = len(dataset)
        self.enable_sampling_record = enable_sampling_record
        self.enable_sampling_negative = enable_sampling_negative
        self.enable_sampling_positive = enable_sampling_positive

        if self.enable_sampling_record:
            self.records_tensors = torch.tensor(dataset, device=self.device)

        if self.enable_sampling_negative:
            self.train_full = torch.sparse_coo_tensor(
                torch.tensor(dataset).permute([1, 0]),
                torch.ones(self.records_len),
                (self.num_user, self.num_item)
            ).to_dense().bool().to(device=self.device)
        if self.enable_sampling_positive:
            assert cache_base is not None and cache_folder is not None
            self.fast_len, self.fast_space = cache_or(cache_base + "fast_space_len.pkl",
                                                      folder=cache_folder,
                                                      generator=lambda: self.generate_fast(title, train_set),
                                                      abort_if_not_exist=in_grid)

            self.fast_len = torch.from_numpy(self.fast_len).to(device=self.device)
            self.fast_space = torch.from_numpy(self.fast_space).to(device=self.device)
    def generate_fast(self, title, train_set):
        fast_space = np.zeros([self.num_user, self.num_item], dtype=int)
        fast_len = np.zeros([self.num_user], dtype=int)
        tx = range(self.num_user)
        if title is not None:
            tx = tqdm(tx, desc=title)
        for u in tx:
            if u not in train_set:
                train_set[u] = {random.randint(0, self.num_item - 1)}
            ngs = train_set[u]
            fast_len[u] = len(ngs)
            fast_space[u] = sorted(list(ngs)) + [-1] * (self.num_item - fast_len[u])

        fast_space = fast_space[:, :max(fast_len)]
        return fast_len, fast_space

    def sampling_record(self, batch_size):
        if not self.enable_sampling_record:
            raise NotImplementedError(f"self.sampling_record: {self.enable_sampling_record}")

        pri = torch.randint(0, self.records_len, [batch_size], device=self.device)
        return self.records_tensors[pri]

    def sampling_negative(self, users, num=1):
        if num <= 0:
            return self.sampling_negative_single(users)

        res = torch.vstack([
            self.sampling_negative_single(users)
            for _ in range(num)
        ]).permute([1, 0])
        assert res.shape == torch.Size([len(users), num])
        return res

    def sampling_negative_single(self, users):
        if not self.enable_sampling_negative:
            raise NotImplementedError(f"enable_sampling_negative: {self.enable_sampling_negative}")

        nri = torch.randint(0, self.num_item, [len(users)], device=self.device)
        nx = self.train_full[users, nri]
        if nx.sum():
            nri[nx] = self.sampling_negative_single(users[nx])
        return nri

    def sampling_positive(self, users, num, return_mask=False):
        if not self.enable_sampling_positive:
            raise NotImplementedError(f"enable_sampling_positive: {self.enable_sampling_positive}")

        batch_size = users.shape[0]
        assert users.shape == torch.Size([batch_size])
        random_indicator = torch.randint(0, self.num_item * 100, [num, batch_size], device=self.device)
        item_mask = random_indicator % self.fast_len[users]
        res = self.fast_space[users, item_mask].T
        assert torch.all(res != -1)
        assert res.shape == torch.Size([batch_size, num])
        if return_mask:
            c = self.fast_len.cumsum(0).roll(1, 0)
            c[0] = 0
            rs = self.records_tensors[:, 1]
            mask = c[users].unsqueeze(1) + item_mask.T
            assert torch.all(rs[mask] == res)
            return res, mask
        else:
            return res

    def sampling_users_with_proportion(self, num):
        if num <= self.num_user:
            return torch.randint(0, self.num_user, [num], device=self.device)
        repeat = (num * self.fast_len / self.fast_len.sum()).int()
        res = torch.cat([
            torch.tensor(range(self.num_user), device=self.device).repeat_interleave(repeat),
            torch.randint(0, self.num_user, [num - repeat.sum()], device=self.device)
        ])
        assert res.shape == torch.Size([num])
        return res
