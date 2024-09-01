
import re
from torchvision import transforms
from typing import List, Dict
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from torchvision.transforms import functional as F  # Import transforms.functional as F
import cv2
import random

import csv
import os
import pandas as pd


import re

from torchvision import transforms

from typing import List, Dict
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from torchvision.transforms import functional as F  # Import transforms.functional as F
import cv2
import random

import csv
import os
import pandas as pd

class LoadImage:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key == 'image path':
                sample[key] = cv2.imread(sample[key], cv2.IMREAD_COLOR)
            elif key == 'target path':
                sample[key] = cv2.imread(sample[key], cv2.IMREAD_GRAYSCALE)

            # Check if the image was loaded correctly
            if sample[key] is None:
                raise ValueError(f"Failed to load image or mask: {sample[key]}")

        return sample


class ResizeImages:
    def __init__(self, size):
        self.size = size  # size should be (height, width)

    def __call__(self, sample):
        # Resize the image with bilinear interpolation
        if 'image path' in sample:
            sample['image path'] = cv2.resize(sample['image path'], (self.size[1], self.size[0]),
                                              interpolation=cv2.INTER_LINEAR)

        # Resize the mask with nearest-neighbor interpolation
        if 'target path' in sample:
            sample['target path'] = cv2.resize(sample['target path'], (self.size[1], self.size[0]),
                                               interpolation=cv2.INTER_NEAREST)

        return sample


class ToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key not in sample:
                raise KeyError(f"Key '{key}' not found in the sample")

            image = sample[key]

            # Convert the 'image path' key using F.to_tensor
            if key == 'image path' and isinstance(image, Image.Image):
                sample[key] = F.to_tensor(image)

            # Convert the 'target path' key using torch.tensor
            elif key == 'target path':
                if isinstance(image, Image.Image):
                    sample[key] = torch.tensor(np.array(image), dtype=torch.long)
                elif isinstance(image, np.ndarray):
                    sample[key] = torch.tensor(image, dtype=torch.long)

        return sample


class NormalizeImages:
    def __init__(self, keys, mean, std):
        self.keys = keys
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = (sample[key].astype(np.float32) / 255.0 - self.mean) / self.std
        return sample


class RandomAugmentations:
    def __init__(self, keys, max_rotation_angle=30):
        self.keys = keys
        self.max_rotation_angle = max_rotation_angle

    def __call__(self, sample):
        # Random decision for horizontal flip
        if random.random() > 0.5:
            for key in self.keys:
                sample[key] = cv2.flip(sample[key], 1)  # Apply horizontal flip

        # Random decision for vertical flip
        if random.random() > 0.5:
            for key in self.keys:
                sample[key] = cv2.flip(sample[key], 0)  # Apply vertical flip

        # Random decision for rotation
        if random.random() > 0.5:
            # Generate a random rotation angle between -max_rotation_angle and +max_rotation_angle
            angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            for key in self.keys:
                # Get the image shape
                h, w = sample[key].shape[:2]
                # Compute the rotation matrix
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                # Apply the rotation
                sample[key] = cv2.warpAffine(sample[key], M, (w, h), flags=cv2.INTER_NEAREST)

        return sample


class RandomBrightnessContrast:
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), p=0.5):
        self.brightness_range = brightness_range  # Tuple specifying the brightness adjustment range
        self.contrast_range = contrast_range  # Tuple specifying the contrast adjustment range
        self.p = p  # Probability of applying the transformation

    def __call__(self, sample):
        if random.random() < self.p:
            for key in sample.keys():
                if key == 'image path':  # Only apply to the 'image path' key
                    image = sample[key]

                    # Ensure the image is in uint8 format
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)  # Convert to 8-bit format if necessary

                    # Convert to PIL Image for transformations
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL Image

                    # Random brightness adjustment
                    brightness_factor = np.random.uniform(*self.brightness_range)
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(brightness_factor)

                    # Random contrast adjustment
                    contrast_factor = np.random.uniform(*self.contrast_range)
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(contrast_factor)

                    # Convert back to OpenCV format
                    sample[key] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return sample


class BlendCLAHEandNormalize:
    def __init__(self, keys, mean, std, alpha=0.5, clip_limit=1.5, tile_grid_size=(8, 8)):
        self.keys = keys
        self.mean = mean
        self.std = std
        self.alpha = alpha
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, sample):
        for key in self.keys:
            img = sample[key]

            # Convert the image to uint8 if it's not already in that format
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            if len(img.shape) == 2:  # Grayscale image
                img_clahe = self.clahe.apply(img)
            else:  # Color image, apply CLAHE on each channel separately
                img_clahe = np.stack([self.clahe.apply(channel) for channel in cv2.split(img)], axis=-1)

            img_clahe = img_clahe.astype(np.float32) / 255.0
            img_normalized = (img_clahe - self.mean) / self.std

            # Blend CLAHE-enhanced image with normalized image
            sample[key] = self.alpha * img_clahe + (1 - self.alpha) * img_normalized
        return sample


class NormalizeLabels:
    def __init__(self, keys, ignore_label_id=255):
        self.keys = keys
        self.ignore_label_id = ignore_label_id

    def __call__(self, sample):
        for key in self.keys:
            mask = sample[key]

            # Initialize the mapping array
            mapping = np.arange(256)  # Create a mapping from 0 to 255
            mapping[self.ignore_label_id] = 0  # Map 255 to 0
            mapping[0:20] = np.arange(1, 21)  # Map 0-19 to 1-20

            # Apply the mapping to the mask
            sample[key] = mapping[mask]
        return sample

#########################################################################################

from torchvision import transforms as TF

valid_data_transform = TF.Compose([
    LoadImage(keys=['image path', 'target path']),
    ResizeImages(size=[256, 256]),
    NormalizeLabels(keys=['target path']),
    # BlendCLAHEandNormalize(keys=['image path'], mean=[0.2868955433368683, 0.3251330256462097, 0.28389179706573486], std=[0.18696387112140656, 0.19017396867275238, 0.18720199167728424]),
    NormalizeImages(keys=['image path'], mean=[0.2868955433368683, 0.3251330256462097, 0.28389179706573486],
                    std=[0.18696387112140656, 0.19017396867275238, 0.18720199167728424]),
])

##########################################################################################################
from torch.utils.data import Dataset
class Dataset(Dataset):
    def __init__(self, input_dataframe: pd.DataFrame, root_dir: str, KeysOfInterest: List[str],
                 data_transform: transforms.Compose):
        self.root_dir = root_dir
        self.koi = KeysOfInterest
        self.input_dataframe = input_dataframe[self.koi]
        self.data_transform = data_transform

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Get the row data for the given index
        row = self.input_dataframe.iloc[index]

        # Initialize a dictionary to hold the sample
        sample = {}

        # Loop through each key of interest to load the respective data
        for key in self.koi:
            # Extract and modify the relative path
            relative_path = row[key]
            # Remove the initial part of the path using re.sub
            relative_path = re.sub(r'^.*?data/', '', relative_path)
            # Construct the full path from root directory and modified relative path
            file_path = os.path.join(self.root_dir, relative_path)

            # Check if the file exists before loading
            if os.path.exists(file_path):
                # Open the image file and store it in the sample dictionary
                sample[key] = file_path
            else:
                print(f"File not found: {file_path}")
                # Handle missing file scenario (e.g., set to None or raise an exception)
                sample[key] = None

        # Apply transformations to the sample
        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

    def __len__(self):
        return len(self.input_dataframe)

###################################################################################################
# Assuming you have the CSVs loaded into pandas DataFrames

csv_test = pd.read_csv("test_data.csv")
# Specify the root directory where the images and masks are stored
root_directory = "/home/moustafa/Cityscapes_Moustafa/Mount/data"

# Specify the keys of interest, typically the columns for images and masks in your DataFrame
keys_of_interest = ["image path", "target path"]

# Initialize the training dataset

# Initialize the validation dataset
ds_test = Dataset(
    input_dataframe=csv_test,
    root_dir=root_directory,
    KeysOfInterest=keys_of_interest,
    data_transform=valid_data_transform
)

########################################################################################
from torch.utils.data import DataLoader

# Initialize the DataLoader for the training dataset
dl_test = DataLoader(
    dataset=ds_test,
    batch_size=16,            # Batch size for training
    num_workers=4,           # Number of worker threads to load data       # Number of samples loaded in advance by each worker
    shuffle=True,         # Shuffle the data at every epoch
)
#####################################################################################################################################################



