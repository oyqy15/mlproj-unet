import albumentations as albu


def training_augmentation():
    ar = [
        albu.Resize(320, 640)
    ]
    return albu.Compose(ar)

def validation_augmentation():
    ar = [
        albu.Resize(320, 640)
    ]
    return albu.Compose(ar)

def training_augmentation_kaggle():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=0.5,
            border_mode=0
        ),
        albu.GridDistortion(p=0.5),
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return albu.Compose(train_transform)

def validation_augmentation_kaggle():
    test_transform = [
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return albu.Compose(test_transform)

# why 2 augmentations diff?