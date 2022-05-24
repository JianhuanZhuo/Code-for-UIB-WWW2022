from dataset import ML100KDataset
from tools.config import load_config


class Yelp2018Dataset(ML100KDataset):
    def cfg_name(self):
        return "yelp2018", "data.txt", "\t"


if __name__ == '__main__':
    """
    #user : 31668
    #item : 38048
    #train: 1498070
    #valid: 31668
    #tests: 31668
    """
    cfg = load_config("../config.yaml")
    dataset = Yelp2018Dataset(config=cfg)
