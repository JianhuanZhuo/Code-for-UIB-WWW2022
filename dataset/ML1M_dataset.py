import os

from dataset import ML100KDataset
from tools.config import load_config


class ML1MDataset(ML100KDataset):
    def cfg_name(self):
        return "ML1M", "ratings.dat", "::"


if __name__ == '__main__':
    """
    #user : 6040
    #item : 3706
    #train: 988129
    #valid: 6040
    #tests: 6040
    """
    cfg = load_config("../config.yaml")
    dataset = ML1MDataset(config=cfg)
