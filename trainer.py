import os
import random
import sys
import traceback

import numpy as np
import torch
from setproctitle import setproctitle
from torch.optim.adagrad import Adagrad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataset import dataset_resolver
from evaluator import Evaluator
from model import model_resolver, op_resolver
from tools.config import load_specific_config
from tools.tee import StdoutTee, StderrTee


def wrap(config):
    pid = os.getpid()
    config['pid'] = pid
    if "grid_spec" not in config:
        print(f"pid is {pid}")
    grid_spec = ""
    if "grid_spec" in config:
        total = config.get_or_default("grid_spec/total", -1)
        current = config.get_or_default("grid_spec/current", -1)
        # print(f"grid spec: {current:02}/{total:02} on cuda:{config['cuda']}")
        grid_spec = f"{current:02}/{total:02}/{config['cuda']}#"

    if 'writer_path' not in config or config['writer_path'] is None:
        folder = config['log_tag']
        folder += '-%s' % (config["timestamp_mask"])
        if config["git/state"] == "Good":
            folder += '-%s' % (config['git']['hexsha'][:5])

        config['writer_path'] = os.path.join(config['log_folder'],
                                             folder,
                                             config.postfix()
                                             )
    if not os.path.exists(config['writer_path']):
        os.makedirs(config['writer_path'])

    setproctitle(grid_spec + config['writer_path'])

    if 'logfile' not in config or config['logfile']:
        logfile_std = os.path.join(config['writer_path'], "std.log")
        logfile_err = os.path.join(config['writer_path'], "err.log")
        with StdoutTee(logfile_std, buff=1), StderrTee(logfile_err, buff=1):
            try:
                main_run(config)
            except Exception as e:
                with open(os.path.join(config['writer_path'], "runtime-error.txt"), "w") as fp:
                    fp.write(str(e) + "\n")
                sys.stderr.write(str(e) + "\n")
                print(traceback.print_exc())
                raise e
    else:
        main_run(config)
    torch.cuda.empty_cache()
    return


def main_run(config):
    device = config.getx("device", "cuda")
    if "grid_spec" not in config:
        print(config)
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    if config.get_or_default('cuda', 'auto') == 'auto':
        import GPUtil
        gpus = GPUtil.getAvailable(order='memory', limit=1)
        assert len(gpus) != 0
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpus[0]}"
        print(f"automatically switch to cuda: {gpus[0]}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda']

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    summary = SummaryWriter(config['writer_path'])
    summary.add_text('config', config.__str__())
    if "grid_spec" not in config:
        print(f"output to {config['writer_path']}")

    dataset = dataset_resolver.make(config.getx("dataset/source", None), config=config)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
    )

    # 模型定义
    model = model_resolver.make(config.getx('model/model'), config=config, dataset=dataset)
    if "grid_spec" not in config:
        print("loading model and assign GPU memory...")
        model = model.to(device=device)
        print("loaded over.")
    else:
        model = model.to(device=device)

    evaluator = Evaluator(config, summary, dataset)
    # 优化器
    if config.getx("model/optimizer_class", "Adagrad") == "Adagrad":
        optimizer = Adagrad(model.parameters(), **config['model/optimizer'])
    else:
        optimizer_class = op_resolver.lookup(config.getx("model/optimizer_class", "Adagrad"))
        optimizer = optimizer_class(model.parameters(), **config['model/optimizer'])

    epoch_loop = range(config['epochs'])
    if config.get_or_default("train/epoch_tqdm", False):
        epoch_loop = tqdm(epoch_loop,
                          desc="train",
                          bar_format="{desc} {percentage:3.0f}%|{bar:10}|{n_fmt:>5}/{total_fmt} "
                                     "[{elapsed}<{remaining} {rate_fmt}{postfix}]",
                          )
    for epoch in epoch_loop:
        # 数据记录和精度验证
        if epoch % config['evaluator_time'] == 0:
            if config.getx("evaluator_args/checkpoint_save", False):
                torch.save({
                    "model": model.state_dict(),
                    "config": config,
                }, os.path.join(config['writer_path'], f"checkpoint-{epoch:04}.tar"))
            evaluator.evaluate(model, epoch)

            if evaluator.should_stop():
                if "grid_spec" not in config:
                    print("early stop...")
                break
        # 我们 propose 的模型训练
        epoch_loss = []
        loader = dataloader
        if config.get_or_default("train/batch_tqdm", True):
            loader = tqdm(loader,
                          desc=f'train  \tepoch: {epoch:05}/{config["epochs"]}',
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                          )
        for batch, packs in enumerate(loader):
            optimizer.zero_grad()
            model.train()

            loss = model(*packs)
            regular = model.additional_regularization()
            total = loss.sum() + regular

            total.backward()
            optimizer.step()
            epoch_loss.append(loss.mean().item())

            assert model.user_embedding.weight.isnan().sum() == 0
            assert model.item_embedding.weight.isnan().sum() == 0

            model.batch_hook(epoch=epoch, batch=batch)

        summary.add_scalar('Epoch/Loss', np.mean(epoch_loss), global_step=epoch)
        if isinstance(epoch_loop, tqdm):
            if "grid_spec" in config:
                total = config.get_or_default("grid_spec/total", -1)
                current = config.get_or_default("grid_spec/current", -1)
                epoch_loop.desc = f"{current:04}/{total:04}/{config['cuda']}#"
            else:
                epoch_loop.desc = f"train {np.mean(epoch_loss):5.4f}"

        model.epoch_hook(epoch=epoch)

    evaluator.record_best()
    summary.close()
    if config.getx("evaluator_args/best_save", False):
        torch.save({
            "model": model.state_dict(),
            "config": config,
        }, os.path.join(config['writer_path'], f"checkpoint-best.tar"))


if __name__ == '__main__':
    cfg = load_specific_config("configs/ML1M/LGN.yaml")
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print("additional argument: " + arg)
            if "=" in arg and len(arg.split("=")) == 2:
                k, v = arg.strip().split("=")
                if v.lower() in ['false', 'no', 'N', 'n']:
                    v = False
                elif v.lower() in ['true', 'yes', 'Y', 'y']:
                    v = True
                else:
                    try:
                        v = eval(v)
                    except ValueError:
                        pass
                cfg[k] = v
                continue
            print("arg warning : " + arg)
            exit(0)
    wrap(cfg)
