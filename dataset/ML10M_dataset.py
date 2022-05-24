from dataset import ML100KDataset
from tools.config import load_config


class ML10MDataset(ML100KDataset):
    def cfg_name(self):
        return "ML10M", "ratings.dat", "::"


if __name__ == '__main__':
    """
    #user : 69878
    #item : 10677
    #train: 9860298
    #valid: 69878
    #tests: 69878
    """
    cfg = load_config("../config.yaml")
    dataset1 = ML10MDataset(config=cfg)
    # dataset2 = ML10MDataset(config=cfg)
    # dataset2 = ML10MDataset(config=cfg)
    # dataset2 = ML10MDataset(config=cfg)
