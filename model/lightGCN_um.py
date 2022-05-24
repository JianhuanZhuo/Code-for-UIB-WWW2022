import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Embedding

from model import BasedModel


class GCNLayer(nn.Module):
    def __init__(self, graph, out_d, in_dg):
        super(GCNLayer, self).__init__()
        self.graph = graph
        self.out_d = out_d
        self.in_dg = in_dg

    def forward(self, node_f):
        with self.graph.local_scope():
            node_f = node_f * self.out_d

            self.graph.ndata['n_f'] = node_f
            self.graph.update_all(message_func=fn.copy_src(src='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = self.graph.ndata['n_f']

            rst = rst * self.in_dg

            return rst


class UmLGNModel(BasedModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.n_layers = self.config['model/n_layers']
        self.drop = self.config['model/drop']
        self.decay = self.config['model/decay']

        self.inter_graph = dgl.graph((torch.tensor([x for x, y in dataset.train]),
                                      torch.tensor([self.num_user + y for x, y in dataset.train])),
                                     num_nodes=self.num_user + self.num_item,
                                     idtype=torch.int32)
        self.inter_graph = dgl.to_bidirected(self.inter_graph).to(device=self.device)

        with torch.no_grad():
            out_d = self.inter_graph.out_degrees().float().clamp(min=1).pow(-0.5).view(-1, 1)
            in_dg = self.inter_graph.in_degrees().float().clamp(min=1).pow(-0.5).view(-1, 1)

        self.layers = nn.ModuleList([GCNLayer(self.inter_graph, out_d, in_dg) for _ in range(self.n_layers)])

        self.mode = self.config.getx("model/mode", 'linear')
        if self.mode == 'linear':
            self.um = Linear(self.dim, 1)
        elif self.mode == 'embed':
            self.um = Embedding(self.num_user, 1)
        else:
            raise NotImplementedError(f"model/mode : {self.mode}")
        self.alpha = self.config.getx("model/alpha", 0.5)

    def compute_graph(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = F.dropout(torch.cat([users_emb, items_emb], dim=0), p=self.drop)
        light_out = all_emb

        for i, layer in enumerate(self.layers):
            all_emb = layer(all_emb)
            light_out = light_out + all_emb
        light_out = light_out / (len(self.layers) + 1)
        users, items = torch.split(light_out, [self.num_user, self.num_item])
        return users, items

    def forward(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        assert users.shape == pos_items.shape == neg_items.shape

        users_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.item_embedding(pos_items)
        neg_emb_ego = self.item_embedding(neg_items)
        reg_loss = torch.cat([users_emb_ego, pos_emb_ego, neg_emb_ego]).norm(2).pow(2) / float(batch_size) / 2

        all_users, all_items = self.compute_graph()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        pos_scores = torch.sum(users_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=-1)

        assert pos_scores.shape == torch.Size([batch_size])
        assert neg_scores.shape == torch.Size([batch_size])

        if self.mode == 'linear':
            boundary = self.um(users_emb).reshape([batch_size])
        elif self.mode == 'embed':
            boundary = self.um(users).reshape([batch_size])
        else:
            raise NotImplementedError(f"model/mode : {self.mode}")

        up_loss = F.softplus(boundary - pos_scores)
        un_loss = F.softplus(neg_scores - boundary)

        return (1 - self.alpha) * up_loss + self.alpha * un_loss + reg_loss * self.decay

    def part_distances(self, users, items, **kwargs):
        with torch.no_grad():
            all_users, all_items = self.compute_graph()
            users_emb = all_users[users]
            items_emb = all_items[items]
            rating = torch.sum(users_emb * items_emb, -1)
            return rating
