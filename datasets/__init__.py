from torch.utils.data import DataLoader, Dataset
from utils.transforms import get_transforms
from .kfood import KFoodDataset


def get_dataset(name: str, is_train: bool, split: str = None, **kwargs) -> Dataset:
    datasets_dict = {
        "KFood": KFoodDataset,
    }

    if name not in datasets_dict:
        raise ValueError(f"Dataset '{name}' is not supported. Available: {list(datasets_dict.keys())}")

    transform = get_transforms(name, is_train=is_train)

    resolved_split = split if split is not None else ("train" if is_train else "val")
    return datasets_dict[name](split=resolved_split, transform=transform, **kwargs)


def get_dataloader(dataset: Dataset, batch_size: int, is_train: bool,
                   num_workers: int = 0, **kwargs) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )
