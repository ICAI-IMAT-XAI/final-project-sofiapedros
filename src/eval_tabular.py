import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import accuracy, set_seed, load_model
from src.data import get_dataloaders
import os
from torch.utils.data import DataLoader

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
os.makedirs("./images_tabular", exist_ok=True)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device,
    image_name: str = "confusion_matrix",
) -> tuple:
    """
    Computes accuracy and plots confusion_matrix.
    Args:
    - model (nn.Module): model to evaluate
    - dataloader (Dataloader): dataloader
    - device (torch.device)
    - image_name (str): name for the confusion matrix
    Returns:
    tuple with predictions, targets and softmax outputs
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_softmax = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(outputs)
            all_softmax.append(probs)

            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_softmax = torch.cat(all_softmax)

    acc = accuracy(all_preds, all_targets)

    print(f"Accuracy: {acc*100:.2f}%")

    preds_labels = torch.argmax(all_preds, dim=1).cpu().numpy()
    true_labels = all_targets.cpu().numpy()
    conf_matrix = confusion_matrix(true_labels, preds_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "malignant", "normal"],
        yticklabels=["benign", "malignant", "normal"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"images_tabular/{image_name}.png")

    return preds_labels, true_labels, all_softmax


def main():
    model_name: str = (
        "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_True_drop_True_cw_True_1.5"
    )
    batch_size: int = 32
    seed: int = 42
    info_filename: str = r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    images_and_masks_foldername: str = r"data/BrEaST-Lesions_USG-images_and_masks/"
    drop: bool = True

    set_seed(seed)

    _, _, test_data = get_dataloaders(
        info_filename,
        images_and_masks_foldername,
        batch_size,
        seed=seed,
        type="tabular",
        use_label_encoding=True,
        drop_specific_columns=drop,
    )

    model = load_model(model_name)

    _, _, _ = evaluate_model(
        model, test_data, device, image_name=f"{model_name}_confusion_matrix"
    )


if __name__ == "__main__":
    main()
