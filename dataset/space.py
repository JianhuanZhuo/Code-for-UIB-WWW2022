import numpy as np
from numpy import random

import torch


class Space:
    def __init__(self, num_key, num_value, train_set, num, device, trans=lambda x: x):
        self.num_key = num_key
        self.num_value = num_value
        self.num = num
        self.device = device

        self.fast_space = np.zeros([self.num_key, self.num_value], dtype=int)
        self.fast_len = np.zeros([self.num_key], dtype=int)

        for u in range(self.num_key):
            if u not in train_set:
                # print(f"u not in train_set: {u}")
                train_set[u] = {random.randint(0, num_value - 1)}
            ngs = trans(train_set[u])
            self.fast_len[u] = len(ngs)
            self.fast_space[u] = list(ngs) + [-1] * (self.num_value - self.fast_len[u])

        self.fast_space = self.fast_space[:, :max(self.fast_len)]

        self.fast_len = torch.from_numpy(self.fast_len).to(device=self.device)
        self.fast_space = torch.from_numpy(self.fast_space).to(device=self.device)

    def get_random_indicator(self, batch_size):
        return torch.randint(0, self.num_value * 100, [self.num, batch_size], device=self.device)

    def mod(self, x, y):
        return x % y

    def double_index(self, users, negatives_index):
        return self.fast_space[users, negatives_index]

    def fast_index(self, users, ri):
        mod_index = self.mod(ri, self.fast_len[users])
        return self.double_index(users, mod_index).T

    def sampling(self, users):
        batch_size = users.shape[0]
        assert users.shape == torch.Size([batch_size])
        random_indicator = self.get_random_indicator(batch_size)
        res = self.fast_index(users, random_indicator)
        assert res.shape == torch.Size([batch_size, self.num])
        return res
