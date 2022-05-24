from multiprocessing import Pool
import random

from tools import config
from tools.config import Config
from trainer import wrap
from itertools import product


def grid_search(gpus, config_file):
    process_pool = Pool(len(gpus))
    if isinstance(config_file, str):
        exp_config = config.load_specific_config(config_file)
    elif isinstance(config_file, Config):
        exp_config = config_file
    else:
        raise NotImplementedError()

    grid = exp_config['_grid_search_']
    for k in grid.keys():
        assert k in exp_config['_key_'], f'Search Key not found in _key_: {k}'

    repeat = exp_config.getx("_grid_search_repeat", 1)
    exp_config['log_folder'] = 'grid'
    task = 0

    search_space = list(product(*[
        vs if isinstance(vs, list) else eval(f'{vs}')
        for vs in grid.values()
    ]))
    random.shuffle(search_space)

    total = repeat * len(search_space)
    # assert total <= len(gpus), f"total <= len(gpus): total {total} : len(gpus) {len(gpus)}"
    exp_config['grid_spec/total'] = total
    for r in range(repeat):
        for i, setting in enumerate(search_space):
            # print(setting)
            for idx, k in enumerate(grid.keys()):
                exp_config[k] = setting[idx]
            exp_config['cuda'] = str(gpus[task % len(gpus)])
            task += 1
            exp_config['grid_spec/current'] = task
            process_pool.apply_async(wrap, args=(exp_config.clone(),))
        exp_config.random_again()

    process_pool.close()
    process_pool.join()


if __name__ == '__main__':
    print("please use the dl-search or rd-search!")
