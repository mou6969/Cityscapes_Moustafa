from collections import namedtuple

from DataHandling.data_handling import dl_test

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
dataset_images_dir = "/home/moustafa/Cityscapes_Moustafa/Mount/data"


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


def evaluate_model_on_test_set(dl_test, model, device, num_classes=20, output_dir='Mount/results'):
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
checkpoint_path = "Mount/models/model_checkpoint_epoch_55.pth"

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

# Call this function after loading the model and dataloader
evaluate_model_on_test_set(dl_test, model, device)