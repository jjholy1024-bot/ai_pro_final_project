import os
import json
from PIL import Image
from torch.utils.data import Dataset


class KFoodDataset(Dataset):

    def __init__(self, data_dir: str, split: str, transform=None, **kwargs):
        # TODO: 아래는 데이터셋 초기화 코드입니다. 필요한 경우 수정하세요.
        assert split in ('train', 'val', 'test'), f"split must be train/val/test, got '{split}'"

        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        classes_path = os.path.join(data_dir, 'classes.json')
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_to_idx = json.load(f)

        self.classes = sorted(self.class_to_idx.keys(), key=lambda k: self.class_to_idx[k])

        split_path = os.path.join(data_dir, 'splits', f'{split}.txt')
        self.samples = self._load_split(split_path)

    def _load_split(self, split_path: str):
        # TODO
        raise NotImplementedError

    def __len__(self):
        # TODO
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO
        raise NotImplementedError
