import torch
from torch.nn import functional as F, Dropout, Linear, Embedding

from model import BasedModel, BPRModel


class UmBPRModel(BPRModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.drop = Dropout(self.config.getx("model/drop", 0))
        self.decay = self.config['model/decay']
        # self.um = Embedding(self.num_user, 1)
        # self.um = Linear(self.dim, 1)

        self.mode = self.config.getx("model/mode", 'linear')
        if self.mode == 'linear':
            self.um = Linear(self.dim, 1)
        elif self.mode == 'embed':
            self.um = Embedding(self.num_user, 1)
        else:
            raise NotImplementedError(f"model/mode : {self.mode}")

        self.alpha = self.config.getx("model/alpha", 0.5)

    def forward(self, user, positive, negative, nbs=None, pos_nbs=None, neg_nbs=None):
        assert nbs is None and pos_nbs is None and neg_nbs is None
        batch_size = user.shape[0]
        assert user.shape == positive.shape == negative.shape == torch.Size([batch_size])
        u_emb = self.drop(self.user_embedding(user))
        p_emb = self.drop(self.item_embedding(positive))
        n_emb = self.drop(self.item_embedding(negative))

        reg_loss = torch.cat([u_emb, p_emb, n_emb]).norm(2).pow(2) / float(batch_size) / 2

        up_score = torch.sum(u_emb * p_emb, dim=-1)
        un_score = torch.sum(u_emb * n_emb, dim=-1)
        assert up_score.shape == torch.Size([batch_size])
        assert un_score.shape == torch.Size([batch_size])

        # boundary = self.um(user).reshape([batch_size])
        # boundary = self.um(u_emb).reshape([batch_size])
        if self.mode == 'linear':
            boundary = self.um(u_emb).reshape([batch_size])
        elif self.mode == 'embed':
            boundary = self.um(user).reshape([batch_size])
        else:
            raise NotImplementedError(f"model/mode : {self.mode}")

        up_loss = F.softplus(boundary - up_score)
        un_loss = F.softplus(un_score - boundary)

        return (1 - self.alpha) * up_loss + self.alpha * un_loss + reg_loss * self.decay

    def part_distances(self, users, items, nbs=None, item_nbs=None):
        assert nbs is None and item_nbs is None
        with torch.no_grad():
            user_embed = self.user_embedding(users)
            item_embed = self.item_embedding(items)
            return torch.sum(user_embed * item_embed, dim=-1)

    def additional_regularization(self):
        return 0

    def epoch_hook(self, epoch):
        pass

    def batch_hook(self, epoch, batch):
        pass
