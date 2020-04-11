import os

class DatasetCatalog:
    DATA_DIR = 'Dataset'
    DATASETS = {
        'div8k_train': {
            'data_dir': 'DIV8K/train'
        },
        'div8k_val': {
            'data_dir': 'DIV8K/val'
        },
        'div8k_minival': {
            'data_dir': 'DIV8K/minival'
        },
        'div8k_minitrain': {
            'data_dir': 'DIV8K/minitrain'
        },
        'div8k_test': {
            'data_dir': 'DIV8K/test3'
        },
        'div2k': {
            'data_dir': 'DIV2K'
        },
        'set5': {
            'data_dir': 'Set5'
        },
        'set14': {
            'data_dir': 'Set14'
        },
    }

    @staticmethod
    def get(name):
        if 'div2k' in name:
            div2k_root = DatasetCatalog.DATA_DIR
            if 'DIV2K_ROOT' in os.environ:
                div2k_root = os.environ['DIV2K_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(div2k_root, attrs['data_dir'])
            )
            return dict(factory='TrainDataset', args=args)

        if 'div8k_train' in name:
            div8k_root = DatasetCatalog.DATA_DIR
            if 'DIV8K_TRAIN_ROOT' in os.environ:
                div8k_root = os.environ['DIV8K_TRAIN_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(div8k_root, attrs['data_dir'])
            )
            return dict(factory='TrainDataset', args=args)

        if 'div8k_val' in name:
            div8k_root = DatasetCatalog.DATA_DIR
            if 'DIV8K_VAL_ROOT' in os.environ:
                div8k_root = os.environ['DIV8K_VAL_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(div8k_root, attrs['data_dir'])
            )
            return dict(factory='ValDataset', args=args)

        if 'div8k_minival' in name:
            div8k_root = DatasetCatalog.DATA_DIR
            if 'DIV8K_MINIVAL_ROOT' in os.environ:
                div8k_root = os.environ['DIV8K_MINIVAL_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(div8k_root, attrs['data_dir'])
            )
            return dict(factory='ValDataset', args=args)

        if 'div8k_minitrain' in name:
            div8k_root = DatasetCatalog.DATA_DIR
            if 'DIV8K_MINIVAL_ROOT' in os.environ:
                div8k_root = os.environ['DIV8K_MINITRAIN_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(div8k_root, attrs['data_dir'])
            )
            return dict(factory='MiniTrainDataset', args=args)

        if 'div8k_test' in name:
            div8k_root = DatasetCatalog.DATA_DIR
            if 'DIV8K_TEST_ROOT' in os.environ:
                div8k_root = os.environ['DIV8K_TEST_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(div8k_root, attrs['data_dir'])
            )
            return dict(factory='ValDataset', args=args)

        if 'set5' in name:
            set5_root = DatasetCatalog.DATA_DIR
            if 'Set5_ROOT' in os.environ:
                set5_root = os.environ['Set5_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(set5_root, attrs['data_dir'])
            )
            return dict(factory='TrainDataset', args=args)

        if 'set14' in name:
            set14_root = DatasetCatalog.DATA_DIR
            if 'Set14_ROOT' in os.environ:
                set14_root = os.environ['Set14_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(set14_root, attrs['data_dir'])
            )
            return dict(factory='TrainDataset', args=args)
            

        else:
            raise RuntimeError('Dataset not available: {}'.format(name))
