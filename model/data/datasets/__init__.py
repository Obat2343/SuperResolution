from torch.utils.data import ConcatDataset

from model.config.path_catlog import DatasetCatalog
from .sr_dataset import TrainDataset, ValDataset, MiniTrainDataset



_DATASETS = {
    'TrainDataset': TrainDataset,
    'ValDataset': ValDataset,
    'MiniTrainDataset': MiniTrainDataset,
}

def build_dataset(dataset_list, transform=None, edge=None):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['edge'] = edge
        dataset = factory(**args)
        datasets.append(dataset)

    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    return dataset