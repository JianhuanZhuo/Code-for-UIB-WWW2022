from dataset import ML1MDataset, LastFMDataset, AIVDataset
from dataset.ML10M_dataset import ML10MDataset
from tools.config import load_config

if __name__ == '__main__':
    """
    #user : 6034
    #item : 3533
    #train: 457813
    #valid: 57250
    #tests: 60209
    """
    cfg = load_config("config.yaml")
    # ML10MDataset(config=cfg)
    ML1MDataset(config=cfg)
    LastFMDataset(config=cfg)
    AIVDataset(config=cfg)
