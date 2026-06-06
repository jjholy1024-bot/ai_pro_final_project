import os
import json
import hashlib
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class KFoodDataset(Dataset):

    def __init__(self, data_dir: str, split: str, transform=None, cleanse_data: bool = True, **kwargs):
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
        
        if cleanse_data and split != 'test':
            self.samples = self._cleanse_samples(self.samples)

    def _load_split(self, split_path: str):
        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if self.split == 'test':
                    # test.txt: test_images/test_00000
                    img_path = os.path.join(self.data_dir, f"{line}.jpg")
                    samples.append((img_path, -1)) # dummy label
                else:
                    # train.txt / val.txt: 구이류/갈치구이/Img_001_0001 0
                    rel_path, label_str = line.rsplit(' ', 1)
                    img_path = os.path.join(self.data_dir, 'images', f"{rel_path}.jpg")
                    samples.append((img_path, int(label_str)))
        return samples

    def _cleanse_samples(self, samples):
        """
        MD5 해시를 기반으로 중복 및 레이블 오류 데이터를 제거합니다.
        1. 동일 이미지 + 동일 레이블: 중복 제거 (1개만 유지)
        2. 동일 이미지 + 다른 레이블: 레이블 오류로 판단하여 해당 이미지 모두 제거
        """
        print(f"[{self.split}] Cleansing data based on MD5 hashes...")
        md5_to_samples = {}
        for img_path, label in tqdm(samples, desc="Calculating MD5"):
            with open(img_path, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            
            if md5 not in md5_to_samples:
                md5_to_samples[md5] = []
            md5_to_samples[md5].append((img_path, label))
        
        cleansed_samples = []
        removed_duplicates = 0
        removed_label_errors = 0
        
        for md5, sample_list in md5_to_samples.items():
            labels = set([s[1] for s in sample_list])
            
            if len(labels) > 1:
                # 클래스 간 중복 (레이블 오류): 모두 제거
                removed_label_errors += len(sample_list)
                continue
            
            # 클래스 내 중복: 1개만 유지
            cleansed_samples.append(sample_list[0])
            if len(sample_list) > 1:
                removed_duplicates += (len(sample_list) - 1)
        
        print(f"[{self.split}] Original: {len(samples)} | Cleansed: {len(cleansed_samples)}")
        print(f"  - Removed {removed_duplicates} intra-class duplicates")
        print(f"  - Removed {removed_label_errors} inter-class label errors")
        
        return cleansed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label
