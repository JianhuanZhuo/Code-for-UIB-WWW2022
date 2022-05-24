from abc import ABC

import torch
from torch.nn import Module, Embedding


class BasedModel(Module, ABC):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.device = config.getx("device", "cuda")
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.num_train = config['num_train']
        self.dim = config['model/dim']

        self.user_embedding = Embedding(self.num_user, self.dim)
        self.item_embedding = Embedding(self.num_item, self.dim)

        self.return_neighbors = self.config.getx("dataset/return_neighbors", False)
        self.neighbors_num = config.getx("dataset/neighbors_num", 1)

        # def shape_check(module, fea_in, fea_out):
        #     if self.return_neighbors:
        #
        #         user, positive, negative, nbs = fea_in
        #         batch_size = user.shape[0]
        #         assert user.shape == torch.Size([batch_size])
        #         assert positive.shape == torch.Size([batch_size])
        #         assert nbs.shape == torch.Size([batch_size, self.neighbors_num])
        #
        #         assert fea_out.shape == torch.Size([batch_size]), f"shape: {fea_out.shape} != [{batch_size}]"
        #     else:
        #         user, positive, negative = fea_in
        #         batch_size = user.shape[0]
        #         assert user.shape == torch.Size([batch_size])
        #         assert positive.shape == torch.Size([batch_size])
        #
        #         assert fea_out.shape == torch.Size([batch_size]), f"shape: {fea_out.shape} != [{batch_size}]"
        #
        #     return None
        #
        # self.register_forward_hook(hook=shape_check)

    def part_distances(self, users, items, **kwargs):
        raise NotImplementedError()

    def additional_regularization(self):
        return 0

    def epoch_hook(self, epoch):
        pass

    def batch_hook(self, epoch, batch):
        pass
