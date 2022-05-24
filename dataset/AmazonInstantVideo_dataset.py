from dataset import ML100KDataset
from tools.config import load_config


class AIVDataset(ML100KDataset):
    def cfg_name(self):
        return "AmazonInstantVideo", "Amazon_Instant_Video_5.txt", "\t"


if __name__ == '__main__':
    """
    #user : 5130
    #item : 1685
    #train: 26866
    #valid: 5130
    #tests: 5130
    """
    cfg = load_config("../config.yaml")
    dataset = AIVDataset(config=cfg)
