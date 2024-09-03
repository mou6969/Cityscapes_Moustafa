from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , 34 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
num_classes = len(set(label.trainId for label in labels if label.trainId != 255))

# Print number of classes
print(f"Number of classes: {num_classes}")


##########################################################################################################


#################################################################################################################


#################################################################################################################


#########################################################################################

#########################################################################################################################################################

from Utilities.Utilities import iou_metric, calculate_dice
from UNet.UNet import UNet

import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import csv
import os
import pandas as pd

# Path to the dataset
dataset_images_dir = "data/"


# Paths for CSV files
test_csv_path = "test_data.csv"

# Initialize lists to hold file paths
all_files = []

# Iterate over all subfolders and files in the images directory
for root, dirs, files in os.walk(dataset_images_dir):
    for file_name in files:
        if file_name.endswith("_leftImg8bit.png"):
            all_files.append(os.path.join(root, file_name))


def process_files(file_list, csv_writer):
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        city_prefix = file_name.split("_")[0]

        # Extract the image prefix
        image_prefix = "_".join(file_name.split("_")[1:3])

        # Construct the corresponding label image path
        label_image_name = f"{city_prefix}_{image_prefix}_gtFine_labelTrainIds.png"
        label_image_path = os.path.join(dataset_images_dir,label_image_name)


        if not os.path.isfile(label_image_path):
            print(f"Warning: Label image {label_image_path} not found.")
            continue

        # Write the image and label paths to the CSV file
        csv_writer.writerow([file_path, label_image_path])


# Create CSV files for training and validation data
with open(test_csv_path, mode='w', newline='') as test_file:
    test_writer = csv.writer(test_file)

    # Write headers
    test_writer.writerow(["image path", "target path"])

    # Process all images for training
    process_files(all_files, test_writer)

print("Debug: Finished writing to test_data.csv")




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
root_directory = "/home/moustafa/Cityscapes_Moustafa/data/"

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





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_and_save_samples(inputs, targets, predictions, num_classes, save_path, batch_idx):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert tensors to numpy arrays for visualization
    inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format
    targets_np = targets.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    for i in range(inputs_np.shape[0]):
        # Create a figure to hold the plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Input Image
        img = inputs_np[i]
        axs[0].imshow(img)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        # Ground Truth Mask
        gt_mask = targets_np[i]
        axs[1].imshow(gt_mask, cmap='tab20b', vmin=0, vmax=num_classes - 1)
        axs[1].set_title('Ground Truth Mask')
        axs[1].axis('off')

        # Model Prediction
        pred_mask = predictions_np[i]
        axs[2].imshow(pred_mask, cmap='tab20b', vmin=0, vmax=num_classes - 1)
        axs[2].set_title('Model Prediction')
        axs[2].axis('off')

        # Save the figure
        filename = os.path.join(save_path, f'batch_{batch_idx}_sample_{i + 1}.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid displaying it

    print(f"Saved {inputs_np.shape[0]} sample plots from batch {batch_idx} to '{save_path}'.")


def evaluate_model_on_test_set(dl_test, model, device, num_classes=20, output_dir='results'):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Iterate through the test set with a progress bar
        for batch_idx, batch in tqdm(enumerate(dl_test), total=len(dl_test), desc='Evaluating Model'):
            inputs = batch['image path'].to(device)
            targets = batch['target path'].to(device).squeeze(1).long().to(device)
            inputs = inputs.permute(0, 3, 1, 2).to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.long)

            # Get model predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Collect predictions and targets
            all_preds.append(predicted.cpu())
            all_labels.append(targets.cpu())

            # Plot and save results for a few samples (e.g., first 5 batches)
            if batch_idx < 1:  # Change this number to adjust the number of batches you want to plot
                plot_and_save_samples(inputs, targets, predicted, num_classes, output_dir, batch_idx)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute IoU
    mean_iou, iou_list = iou_metric(all_preds, all_labels, num_classes=num_classes)
    mean_dice, dice_list = calculate_dice(all_preds, all_labels, num_classes=num_classes)

    # Print the results
    print(f"Mean IoU over test set: {mean_iou:.4f}")
    print(f"Mean Dice over test set: {mean_dice:.4f}")
    for cls, iou in iou_list:
        print(f"Class {cls}: IoU = {iou:.4f}")

    for cls, dice in dice_list:
        print(f"Class {cls}: Dice Score = {dice:.4f}")


# Example usage:
model = UNet(in_channels=3, out_channels=20).to(device)
checkpoint_path = "models/model_checkpoint_epoch_55.pth"

# Load model from checkpoint if exists
if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded. Best Mean IoU: 0.4516")
else:
    start_epoch = 0
    best_mean_iou = 0.0
    print("No checkpoint found. Starting evaluation from scratch.")

evaluate_model_on_test_set(dl_test, model, device)