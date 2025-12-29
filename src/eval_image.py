import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import accuracy, set_seed, load_model
from src.data import get_dataloaders

import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs("./images", exist_ok=True)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    image_name: str = "confusion_matrix",
) -> tuple:
    """Computes accuracy and plots confusion_matrix.
    Args:
    - model (nn.Module): model to evaluate
    - dataloader (Dataloader): dataloader
    - device (torch.device)
    - image_name (str): name for the confusion matrix
    Returns:
    tuple with predictions and targets
    """

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            all_preds.append(outputs)
            all_targets.append(labels)

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute accuracy
    acc = accuracy(all_preds, all_targets)
    print(f"Accuracy: {acc*100:.2f}%")

    # Confusion matrix
    preds_labels = torch.argmax(all_preds, dim=1).cpu().numpy()
    true_labels = all_targets.cpu().numpy()
    cm = confusion_matrix(true_labels, preds_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "malignant", "normal"],
        yticklabels=["benign", "malignant", "normal"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"results/{image_name}.png")

    return preds_labels, true_labels


def main():
    model_name: str = (
        "Image_model_lr_0.0005_bs_32_back_resnet18_drop_0.5_unfreeze_False_extrconv_True_augment_True"
    )
    batch_size: int = 32
    seed: int = 42
    info_filename: str = r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    images_and_masks_foldername: str = r"data/BrEaST-Lesions_USG-images_and_masks/"

    set_seed(seed)

    test_data: DataLoader
    _, _, test_data = get_dataloaders(
        info_filename, images_and_masks_foldername, batch_size, seed=seed, type="image"
    )

    model = load_model(model_name)

    _, _ = evaluate_model(
        model, test_data, device, image_name=f"{model_name}_confusion_matrix"
    )


if __name__ == "__main__":
    main()
