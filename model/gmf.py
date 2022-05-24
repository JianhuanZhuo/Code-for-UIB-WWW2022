import torch
from torch.nn import functional as F, Dropout, Linear

from model import BasedModel


class GMFModel(BasedModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.drop = Dropout(self.config.getx("model/drop", 0))
        self.decay = self.config['model/decay']
        self.beta = Linear(config['model/dim'], 1)

    def forward(self, user, positive, negative, nbs=None, pos_nbs=None, neg_nbs=None):
        assert nbs is None and pos_nbs is None and neg_nbs is None
        batch_size = user.shape[0]
        assert user.shape == positive.shape == negative.shape == torch.Size([batch_size])
        u_emb = self.drop(self.user_embedding(user))
        p_emb = self.drop(self.item_embedding(positive))
        n_emb = self.drop(self.item_embedding(negative))

        reg_loss = torch.cat([u_emb, p_emb, n_emb, self.beta.weight]).norm(2).pow(2) / float(batch_size) / 2

        up_score = self.beta(u_emb * p_emb).squeeze(-1)
        un_score = self.beta(u_emb * n_emb).squeeze(-1)
        assert up_score.shape == torch.Size([batch_size])
        assert un_score.shape == torch.Size([batch_size])

        up_score = F.binary_cross_entropy(torch.sigmoid(up_score),
                                          torch.ones_like(positive, dtype=torch.float),
                                          reduction='none')
        un_score = F.binary_cross_entropy(torch.sigmoid(un_score),
                                          torch.zeros_like(negative, dtype=torch.float),
                                          reduction='none')

        return up_score + un_score + reg_loss * self.decay

    def part_distances(self, users, items, nbs=None, item_nbs=None):
        assert nbs is None and item_nbs is None
        with torch.no_grad():
            user_embed = self.user_embedding(users)
            item_embed = self.item_embedding(items)
            return self.beta(user_embed * item_embed).squeeze(-1)

    def additional_regularization(self):
        return 0

    def epoch_hook(self, epoch):
        pass

    def batch_hook(self, epoch, batch):
        pass
