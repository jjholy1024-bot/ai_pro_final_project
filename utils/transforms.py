import torchvision.transforms as T


def get_transforms(dataset_name: str, is_train: bool):
    if dataset_name == "KFood":
        if is_train == True:
            return T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5), # 좌우 대칭 (50%)
                T.RandomApply([
                T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)
                           )], p=0.3),                  # 이동, 크기 조절, 회전 (30%)
                T.RandomApply([
                T.ColorJitter(brightness=0.3, contrast=0.2)
                ], p=0.3),                              # 밝기 및 대비 조절 
                T.RandomApply([
                T.RandomChoice([
                T.RandomAdjustSharpness(sharpness_factor=2.0),
                    ])], p=0.3),

                T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.5, scale=(0.02, 0.2), value='random')
                ])
        
        elif is_train == False:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
                         ])
    raise ValueError(f"Transforms for '{dataset_name}' are not defined.")



