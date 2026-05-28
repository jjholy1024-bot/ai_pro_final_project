import torchvision.transforms as T


def get_transforms(dataset_name: str, is_train: bool):
    if dataset_name == "KFood":
        if is_train == True:
            # TODO
            raise NotImplementedError("KFood train transform을 구현하세요.")
        elif is_train == False:
            # TODO
            raise NotImplementedError("KFood test transform을 구현하세요.")

    raise ValueError(f"Transforms for '{dataset_name}' are not defined.")
