import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.sampling_space import SamplingSpace
from dataset.space import Space
from tools.config import load_config
from tools.utils import cache_or, group_kv


class ML100KDataset(Dataset):
    def __init__(self, config, share_in_search=False):
        self.share_in_search = share_in_search
        self.config = config
        self.raw_folder, self.raw_file, self.sep = self.cfg_name()
        self.folder = os.path.join(os.path.dirname(__file__), "data", self.raw_folder)

        self.negative_num = self.config.getx("dataset/negative_num", 0)
        self.sampling_with_proportion = self.config.getx("dataset/sampling_with_proportion", False)

        self.rating_filter = config.getx("dataset/filter_rating", 4)
        self.candidate_num = config.getx("dataset/candidate_num", 100)
        self.device = config.getx("device", "cuda")
        self.epochs = config.getx("epochs", 500)
        self.batch_size = config.getx("DataLoader/batch_size", 1024)

        self.train, self.valid, self.tests, self.u_set, self.i_set = self.load_data()
        self.num_user = len(self.u_set)
        self.num_item = len(self.i_set)
        self.train_size = len(self.train)

        self.config['num_user'] = self.num_user
        self.config['num_item'] = self.num_item
        self.config['num_train'] = self.train_size

        if "grid_spec" not in self.config:
            self.statics_dataset()

        self.candidates = cache_or(f"cache.{self.raw_folder}-{self.rating_filter}-{self.candidate_num}.candidates.pkl",
                                   folder=self.folder,
                                   generator=lambda: self.generate_candidates(),
                                   abort_if_not_exist="grid_spec" in self.config)

        self.return_neighbors = self.config.getx("dataset/return_neighbors", False)
        self.item_neighbors = self.config.getx("dataset/item_neighbors", False)
        self.neighbors_num = config.getx("dataset/neighbors_num", 0)
        if self.item_neighbors:
            raise NotImplementedError("self.item_neighbors 为 True 的还没测试好")

        sort_str = "-sort_train" if self.sampling_with_proportion else ""
        train_set = cache_or(f"cache.{self.raw_folder}-{self.rating_filter}{sort_str}-train_set.pkl",
                             folder=self.folder,
                             generator=lambda: group_kv(self.train),
                             abort_if_not_exist="grid_spec" in self.config)

        self.sampling = SamplingSpace(self.num_user, self.num_item, self.train,
                                      device=self.device,
                                      train_set=train_set,
                                      cache_base=f"cache.{self.raw_folder}-{self.rating_filter}{sort_str}-",
                                      cache_folder=self.folder,
                                      title="enable_sampling_positive" if "grid_spec" not in self.config else None,
                                      in_grid="grid_spec" in self.config)

    def statics_dataset(self):
        print("#user :", self.num_user)
        print("#item :", self.num_item)
        print("#train:", len(self.train))
        print("#valid:", len(self.valid))
        print("#tests:", len(self.tests))

    def cfg_name(self):
        return "ML100K", "u.data", "\t"

    def load_data(self):
        sort_str = "-sort_train" if self.sampling_with_proportion else ""
        return cache_or(f"cache.{self.raw_folder}-{self.rating_filter}{sort_str}.split.pkl", folder=self.folder,
                        generator=lambda: self.load_raw(),
                        abort_if_not_exist="grid_spec" in self.config)

    def generate_candidates(self):
        train_set = group_kv(self.train)
        all_set = set(range(self.num_item))
        return {
            user: random.choices(list(all_set - train_set[user]),
                                 k=self.candidate_num)
            for user in tqdm(self.u_set, desc='candidates')
        }

    def __len__(self):
        return self.train_size // self.batch_size + 1

    def sample_up_random(self):
        users_positives = self.sampling.sampling_record(self.batch_size)
        users = users_positives[:, 0]
        positives = users_positives[:, 1]
        return users, positives

    def sample_up_with_proportion(self):
        users = self.sampling.sampling_users_with_proportion(self.batch_size)
        positives, pos_mask = self.sampling.sampling_positive(users, 1, return_mask=True)
        positives = positives.squeeze(1)
        pos_mask = pos_mask.squeeze(1)
        return users, (positives, pos_mask)

    def __getitem__(self, _):
        if self.sampling_with_proportion:
            users, positives = self.sample_up_with_proportion()
        else:
            users, positives = self.sample_up_random()

        negatives = self.sampling.sampling_negative(users, num=self.negative_num)

        if not self.return_neighbors:
            return users, positives, negatives
        else:
            nbs = self.sampling.sampling_positive(users, self.neighbors_num)

            if not self.item_neighbors:
                return users, positives, negatives, nbs
            else:
                pos_nbs = self.item_neighbors_space.sampling(positives)
                neg_nbs = self.item_neighbors_space.sampling(negatives)
                return users, positives, negatives, nbs, pos_nbs, neg_nbs

    def load_raw(self):
        rating_file = os.path.join(self.folder, self.raw_file)
        if not os.path.exists(rating_file):
            if os.path.exists(rating_file + ".zip"):
                import zipfile
                with zipfile.ZipFile(rating_file + ".zip", 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(rating_file))
            else:
                raise FileNotFoundError(rating_file)
        with open(rating_file, "r") as fp:
            sps = [line.strip().split(self.sep) for line in fp]
        uis = [(int(x[0]), int(x[1])) for x in sps if len(x) <=2 or float(x[2]) >= self.rating_filter]

        # 确保有 5 条以上
        kvs = group_kv(uis)
        uis = [
            (k, v)
            for k, vs in kvs.items()
            if len(vs) >= 5
            for v in vs
        ]

        u_set = {u for u, i in uis}
        i_set = {i for u, i in uis}
        remap_u = {u: x for x, u in enumerate(u_set)}
        remap_i = {i: x for x, i in enumerate(i_set)}
        uis_remapped = [
            (remap_u[u], remap_i[i])
            for u, i in uis
        ]

        assert self.config.getx("dataset/split_mode") == "leave-one-out"
        train, valid, tests = self.one_out(uis_remapped)
        train.sort()
        return train, valid, tests, set(remap_u.values()), set(remap_i.values()),

    @staticmethod
    def split_proportion(data):
        """
        按数据集中 user 做分割
        """
        train = []
        valid = []
        tests = []

        for u, vs in group_kv(data).items():
            vs = list(vs)
            size = len(vs)
            random.shuffle(vs)
            for v in vs[:int(size * 0.8)]:
                train.append((u, v))
            for v in vs[int(size * 0.8):int(size * 0.9)]:
                valid.append((u, v))
            for v in vs[int(size * 0.9):]:
                tests.append((u, v))

        return train, valid, tests

    @staticmethod
    def one_out(data):
        train = []
        valid = []
        tests = []

        for u, vs in group_kv(data).items():
            vs = list(vs)
            random.shuffle(vs)
            valid.append((u, vs[0]))
            tests.append((u, vs[1]))
            train += [(u, v) for v in vs[2:]]

        return train, valid, tests


if __name__ == '__main__':
    """
    #user : 942
    #item : 1447
    #train: 43929
    #valid: 5497
    #tests: 5949
    """
    cfg = load_config("../config.yaml")
    dataset = ML100KDataset(config=cfg)
