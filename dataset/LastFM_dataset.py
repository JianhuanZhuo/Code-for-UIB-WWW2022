from dataset import ML100KDataset
from tools.config import load_config


class LastFMDataset(ML100KDataset):
    def cfg_name(self):
        return "LastFM", "user_artists.dat", "\t"


if __name__ == '__main__':
    """
    #user : 1877
    #item : 17617
    #train: 89047
    #valid: 1877
    #tests: 1877
    """
    cfg = load_config("../config.yaml")
    dataset = LastFMDataset(config=cfg)
