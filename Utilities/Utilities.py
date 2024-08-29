import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def pixel_accuracy(predictions, targets):
    # Get the predicted class for each pixel
    preds = torch.argmax(predictions, dim=1)  # B x H x W

    # Compare predicted classes with targets
    correct = (preds == targets).float()

    # Sum the number of correct predictions and divide by the total number of pixels
    accuracy = correct.sum() / correct.numel()

    return accuracy.item()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def iou_metric(preds, labels, num_classes, ignore_index=None):
    iou_list = []
    gt_classes = set(torch.unique(labels).tolist())

    with torch.no_grad():
        for cls in range(num_classes):
            if ignore_index is not None and cls == ignore_index:
                continue  # Skip the ignored class

            # Calculate intersection and union
            intersection = torch.sum((preds == cls) & (labels == cls)).item()
            union = torch.sum((preds == cls) | (labels == cls)).item()

            # Compute IoU for the class
            iou = intersection / union if union > 0 else 0
            iou_list.append((cls, iou))  # Store class ID with IoU value
            print(f"Class {cls}: Intersection = {intersection}, Union = {union}, IoU = {iou}")

    # Filter IoU list to only include classes present in ground truth and not ignored
    valid_iou_list = [(cls, iou) for cls, iou in iou_list if cls in gt_classes]

    # Compute mean IoU, excluding classes that are not present in ground truth
    if len(valid_iou_list) > 0:
        mean_iou = sum(iou for _, iou in valid_iou_list) / len(valid_iou_list)
    else:
        mean_iou = 0.0

    return mean_iou, iou_list


def calculate_dice(predictions, targets, num_classes, ignore_class=None):
    dice_list = []
    gt_classes = set(torch.unique(targets).tolist())  # Get classes present in the targets

    with torch.no_grad():
        for cls in range(num_classes):
            if ignore_class is not None and cls == ignore_class:
                continue  # Skip the ignored class

            # Calculate intersection and union
            pred_mask = (predictions == cls).float()
            target_mask = (targets == cls).float()

            intersection = torch.sum(pred_mask * target_mask).item()
            union = torch.sum(pred_mask + target_mask).item()

            # Compute Dice coefficient for the class
            dice = (2. * intersection + 1e-6) / (union + 1e-6)  # Adding epsilon to avoid division by zero
            dice_list.append((cls, dice))  # Store class ID with Dice value

    # Filter Dice list to only include classes present in ground truth and not ignored
    valid_dice_list = [(cls, dice) for cls, dice in dice_list if cls in gt_classes]

    # Compute mean Dice coefficient, excluding classes that are not present in ground truth
    if len(valid_dice_list) > 0:
        mean_dice = sum(dice for _, dice in valid_dice_list) / len(valid_dice_list)
    else:
        mean_dice = 0.0

    return mean_dice, dice_list