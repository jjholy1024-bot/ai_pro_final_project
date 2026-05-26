import torchvision.transforms as T


def get_transforms(dataset_name: str, is_train: bool):
    if dataset_name == "KFood":
        # TODO
        raise NotImplementedError("KFood transform을 구현하세요.")

    raise ValueError(f"Transforms for '{dataset_name}' are not defined.")
