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