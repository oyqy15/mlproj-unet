# Readme

## dependency

python3

```python
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm as tq
import matplotlib.pyplot as plt
import albumentations as albu
import sklearn
```

## run

### dir structure

```
|-- mlproj-unet (this dir)
|-- data
	|-- train.csv
	|-- sample_submission.csv
	|-- train_images
	|-- test_images
```

### train

```shell
> python main.py -help

usage: main.py [-h] [-mid MI] [-e E] [-lr LR] [-bs B]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -mid MI, --model-id MI
                        Load model from a .pt file (default: test.pt)
  -e E, --max-epochs E  Number of epochs (default: 32)
  -lr LR, --learning-rate LR
                        Learning rate (default: 0.005)
  -bs B, --batch-size B
                        Batch size (default: 8)
```


