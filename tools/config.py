import json
import os
import random
import time

import yaml
from git import Repo
from git import cmd as gitCMD
import copy


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_cache = dict()
        if '_key_' not in self:
            raise NotImplementedError("The _key_ not in the Config")

        if 'seed' not in self:
            self['origin_seed'] = "undefined"
            self['seed'] = random.randint(0, 100)
        else:
            self['origin_seed'] = self['seed']
            if not isinstance(self['seed'], int):
                self['seed'] = random.randint(0, 100)
        self['random_seed'] = self['seed']

        # set timestamp mask
        self['timestamp_mask'] = time.strftime("%m%d-%H%M%S", time.localtime())
        self['pid'] = os.getpid()
        if 'git_update' not in kwargs:
            kwargs['git_update'] = True
        self['git'] = current_git_info(kwargs['git_update'])

    def random_again(self, seed=None):
        self['seed_before'] = self['seed']
        if not seed:
            seed = random.randint(0, 100)
        self['seed'] = seed
        self['random_seed'] = self['seed']

    def postfix(self, *args):
        posts = []
        for k, short in sorted(self['_key_'].items()):
            if short is None:
                short = k.split('/')[-1]
            v = self[k]
            if isinstance(v, bool):
                v = "Y" if v else "N"
            s = f"{short}{v}"
            posts.append(s)
        posts.sort()
        if "grid_spec" in self:
            total = self.get_or_default("grid_spec/total", -1)
            current = self.get_or_default("grid_spec/current", -1)
            cuda = self['cuda']
            grid_spec = f"{current:02}-{total:02}-{cuda}#"
            posts = [grid_spec] + posts
        if args is not None:
            posts = [str(x) for x in args] + posts
        posts.append("pid" + str(self["pid"]))

        post = "-".join(posts)

        if 'folder' in self:
            post = os.path.join(self['folder'], post)

        return post

    def json_str(self):
        return json.dumps(self, indent=2)

    def __contains__(self, item):
        if super(Config, self).__contains__(item):
            return True
        elif '/' not in item:
            return False
        else:
            v = self
            for p in item.split('/'):
                if p not in v:
                    return False
                else:
                    v = v[p]
            return True

    def __getitem__(self, item):
        if super(Config, self).__contains__(item):
            return super(Config, self).__getitem__(item)
        elif '/' in item:
            v = self
            for p in item.split('/'):
                assert p in v, f"key not found: {p}"
                v = v[p]
            return v

    def __setitem__(self, key, value):
        self.attr_cache = dict()
        if super(Config, self).__contains__(key) or '/' not in key:
            super(Config, self).__setitem__(key, value)
        else:
            d = self
            for p in key.split("/")[:-1]:
                v = {}
                if p in d:
                    v = d[p]
                    assert isinstance(v, dict)
                d[p] = v
                d = v
            d[key.split("/")[-1]] = value

    def get_or_default(self, key, default=None):
        if key in self.attr_cache:
            return self.attr_cache[key]
        if key in self:
            res = self[key]
        else:
            res = default
        self.attr_cache[key] = res
        return res

    def getx(self, key, default=None):
        return self.get_or_default(key, default)

    def clone(self):
        return copy.deepcopy(self)


def load_config(path):
    """
    Get and parse the configeration file in `path`
    """
    if path.endswith("yaml") or path.endswith("yml"):
        with open(path, "r") as f:
            return Config(**yaml.load(f, Loader=yaml.FullLoader))
    raise Exception(f"config path exception: {path}")


def load_specific_config(path, default_file_path="./config.yaml", git_update=True):
    if path.endswith("yaml") or path.endswith("yml"):
        if not os.path.exists(default_file_path):
            raise Exception(f"file is not exist : {os.path.abspath(default_file_path)}")
        with open(default_file_path, "r") as cm, open(path, "r") as sf:
            sf_config = yaml.load(sf, Loader=yaml.FullLoader)
            cm_config = yaml.load(cm, Loader=yaml.FullLoader)
            res = {}
            res.update(cm_config)
            res.update(sf_config)
            return Config(**res, git_update=git_update)
    raise Exception(f"config path exception: {path}")


def load_from_tensorboard_dir(eph):
    import tensorflow as tf
    ef = [file for file in os.listdir(eph) if "events" in file][0]
    cfg_str = {}
    for event in tf.compat.v1.train.summary_iterator(os.path.join(eph, ef)):
        vs = event.summary.value
        if not len(vs):
            continue
        v = vs[0]
        if v.tag == 'config/text_summary':
            cfg_str = v.tensor.string_val[0].decode()
            break
    cfg_dict = dict(eval(cfg_str))
    cfg_dict['git_update'] = False
    cfg_obj = Config(**cfg_dict)
    return cfg_obj


def current_git_info(git_update=True):
    try:
        g = gitCMD.Git(".")
        if git_update:
            print("git pull...")
            state = g.pull()
            print("git info:" + state)
            assert state[:len('Already')] == 'Already', f"the result of git state: {state}"
        with Repo(".") as repo:
            t = repo.active_branch.commit.committed_date
            return {
                "state": "Good",
                "active_branch": repo.active_branch.name,
                "working_tree_dir": repo.working_tree_dir,
                "timestamp": t,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
                "summary": repo.active_branch.commit.summary,
                "hexsha": repo.active_branch.commit.hexsha,
            }
    except Exception as e:
        return {
            "state": "Invalid",
            "message": str(e.__str__()),
        }


if __name__ == '__main__':
    # print(current_git_info())
    config = load_config('test.yaml')

    print('pipeline' in config)
    print('pipeline/No' in config)
    print('pipeline/dataset' in config)
    print('No/No' in config)
