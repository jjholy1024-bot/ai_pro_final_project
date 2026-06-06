import torchvision.transforms as T

def get_transforms(dataset_name: str, is_train: bool):
    if dataset_name == "KFood":
        if is_train == True:
            return T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # 색감/대비 강화
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.1) # 가벼운 규제
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
            ])
    raise ValueError(f"Transforms for '{dataset_name}' are not defined.")
