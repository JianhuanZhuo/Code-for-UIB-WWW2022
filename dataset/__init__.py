from class_resolver import Resolver
from torch.utils.data import Dataset

from dataset.ML100K_dataset import ML100KDataset
from dataset.ML1M_dataset import ML1MDataset
from dataset.ML10M_dataset import ML10MDataset
from dataset.LastFM_dataset import LastFMDataset
from dataset.AmazonInstantVideo_dataset import AIVDataset
from dataset.Yelp2018_dataset import Yelp2018Dataset

dataset_resolver = Resolver(
    {
        ML100KDataset,
        ML1MDataset,
        ML10MDataset,
        LastFMDataset,
        AIVDataset,
        Yelp2018Dataset,
    },
    base=Dataset,  # type: ignore
    default=ML100KDataset,
    suffix='dataset',
)
