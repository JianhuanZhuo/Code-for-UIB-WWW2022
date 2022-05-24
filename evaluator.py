import torch
import numpy as np


class Evaluator:

    def __init__(self, config, summary, dataset):
        self.config = config
        self.summary = summary
        self.dataset = dataset
        self.device = config.getx("device", "cuda")

        assert dataset.num_user == len(dataset.valid)
        assert dataset.num_user == len(dataset.tests)
        assert dataset.num_user == len(dataset.candidates)

        self.low_first = config.get_or_default('evaluator_args/low_first', True)
        self.neighbors_num = config.getx("dataset/neighbors_num", 1)

        self.valid_indices = [
            [u for u, i in dataset.valid],
            [i for u, i in dataset.valid]
        ]

        self.tests_indices = [
            [u for u, i in dataset.tests],
            [i for u, i in dataset.tests]
        ]

        self.candi_indices = [
            [u for u, cs in dataset.candidates.items() for c in cs],
            [c for u, cs in dataset.candidates.items() for c in cs]
        ]

        self.select_indices = [
            self.valid_indices[0] + self.tests_indices[0] + self.candi_indices[0],
            self.valid_indices[1] + self.tests_indices[1] + self.candi_indices[1]
        ]
        self.select_tensor = torch.tensor(self.select_indices, device=self.device)

        self.stop_delay = config.get_or_default('evaluator_args/stop_delay', 10)
        self.score_cache = [-np.inf]
        self.best_test_performance = []

        self.return_neighbors = self.config.getx("dataset/return_neighbors", False)
        self.item_neighbors = self.config.getx("dataset/item_neighbors", False)
        if self.return_neighbors:
            self.test_neighbors = self.dataset.sampling.sampling_positive(self.select_tensor[0], self.neighbors_num)

            if self.item_neighbors:
                self.test_neighbors_pos = self.dataset.item_neighbors_space.sampling(self.select_tensor[1])

    def evaluate(self, model, epoch):
        # 失能验证
        if self.config.getx("eval_disable", False):
            return
        model.eval()
        with torch.no_grad():
            xs = self.config.getx("evaluator_args/eval_xs", [1, 3, 5, 10])

            if self.return_neighbors:
                if self.item_neighbors:
                    distances = model.part_distances(self.select_tensor[0],
                                                     self.select_tensor[1],
                                                     self.test_neighbors,
                                                     self.test_neighbors_pos
                                                     )
                else:
                    distances = model.part_distances(self.select_tensor[0], self.select_tensor[1], self.test_neighbors)
            else:
                distances = model.part_distances(self.select_tensor[0], self.select_tensor[1])

            valid_rank, tests_rank = self.rank_position(distances)
            this_is_best = False
            valid_ndcg = (1 / torch.log2((valid_rank[valid_rank < 10] + 2))).sum().item() / self.dataset.num_user
            if valid_ndcg > max(self.score_cache):
                self.score_cache = [valid_ndcg]
                this_is_best = True
            else:
                self.score_cache.append(valid_ndcg)

            ranks = tests_rank
            assert ranks.shape == torch.Size([self.dataset.num_user])
            # assert ranks.min().item() == 0
            assert ranks.max().item() <= 100
            evals = [f"{valid_ndcg:5.4f}"]

            if self.summary:
                self.summary.add_scalar(f"valid/ndcg@10", valid_ndcg, global_step=epoch)

            test_performance = []

            for x in xs:
                hitx = (ranks < x).float().mean().item()
                # 加2因为，这里的排名是 从0 开始的，不是 1
                ndcg = (1 / torch.log2((ranks[ranks < x] + 2))).sum().item() / self.dataset.num_user
                MRRx = (1 / (ranks[ranks < x] + 1)).sum().item() / self.dataset.num_user
                test_performance.append((f"hit@{x}", hitx))
                test_performance.append((f"ndcg@{x}", ndcg))
                test_performance.append((f"MRR@{x}", MRRx))

                # only display @5 and @10 during training
                if x == 1:
                    evals.append(f"{x}: {hitx:5.4f}")
                elif x == 10:
                    evals.append(f"{x}: {hitx:5.4f}/{ndcg:5.4f}/{MRRx:5.4f}")

            if this_is_best:
                evals.append("Best")
                self.best_test_performance = test_performance
            else:
                evals.append(str(len(self.score_cache)))

            if self.config.getx("train/print_eval", False):
                print(f" Eval {epoch:4} | {' | '.join(evals)}", flush=True)

            if self.summary:
                for label, value in test_performance:
                    self.summary.add_scalar("eval/" + label, value, global_step=epoch)

    def rank_position(self, distances):
        assert distances.shape == torch.Size([self.dataset.num_user * 102])

        valid = distances[:self.dataset.num_user].reshape([self.dataset.num_user, 1])
        tests = distances[self.dataset.num_user:self.dataset.num_user * 2].reshape([self.dataset.num_user, 1])
        candi = distances[self.dataset.num_user * 2:].reshape([self.dataset.num_user, 100])

        # the number of topK of negative that bigger than positive is the rank of positive
        # Large is good
        valid_rank = (candi >= valid).sum(dim=-1)
        tests_rank = (candi >= tests).sum(dim=-1)

        return valid_rank, tests_rank

    def should_stop(self):
        if self.config.get_or_default("evaluator_args/use_stop", False):
            return len(self.score_cache) >= self.stop_delay
        return False

    def record_best(self):
        print_str = []
        for label, value in self.best_test_performance:
            if self.summary:
                self.summary.add_scalar("best/" + label, value, global_step=0)
            print_str.append(f"{value:5.4f}")
        s = '|'.join(print_str[2:])
        if self.config.getx("train/print_best", False):
            _grid_thread_hit1 = self.config.getx("_grid_thread_hit1", None)
            if "grid_spec" in self.config and _grid_thread_hit1 is not None:
                hit1 = self.best_test_performance[0][1]
                if hit1 < _grid_thread_hit1:
                    return
                else:
                    s = "\a" + s
            print(f" Best {s}")


if __name__ == '__main__':
    pass
