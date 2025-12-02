import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight

# other libraries
from tqdm.auto import tqdm
from typing import Final
import numpy as np

from src.data import get_dataloaders
from src.utils import save_model, set_seed
from src.models import BreastMLPClassifier
from src.train_functions import train_step, val_step, test_step


def compute_custom_class_weights(
    train_loader: DataLoader, device: torch.device, malignant_penalty: float = 3.0
) -> list:
    """
    Computes custom class weights.
    Increases the weight of the 'malignant' class to further penalize false negatives.

    Args:
    - train_loader (Dataloader): Training DataLoader
    - device (torch.device): device (cuda/cpu)
    - malignant_penalty (float): multiplier factor for the malignant class (>1 to penalize more)

    Returns:
        torch.Tensor containing the weights for each class
    """
    # Collect all labels from the training set
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)

    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)

    # Class weights with inverse frequency
    class_weights = total_samples / (len(unique_classes) * class_counts)

    # Identify the index of the malignant class
    # Assuming: 0=benign, 1=malignant, 2=normal
    # Adjust according to your actual encoding
    malignant_idx = 1

    # Increase the weight of the malignant class
    class_weights[malignant_idx] *= malignant_penalty

    print(class_weights)
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def main():
    epochs: int = 40
    lr: float = 1e-4
    batch_size: int = 8
    dropout: float = 0.2
    hidden_dims: list = (128, 256)
    use_encoding: bool = True
    drop_columns: bool = True
    custom_weights: bool = True
    malignant_penalty: float = 5.0

    info_filename: str = r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    images_folder: str = r"data/BrEaST-Lesions_USG-images_and_masks/"

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Create TensorBoard run
    name: str = (
        f"Tab_model_lr_{lr}_bs_{batch_size}_hd_{str(hidden_dims)}_dropout_{dropout}_encode_{use_encoding}_drop_{drop_columns}_cw_{custom_weights}_{malignant_penalty}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # Load segmentation data
    train_loader, val_loader, test_loader = get_dataloaders(
        info_file=info_filename,
        images_folder=images_folder,
        batch_size=batch_size,
        type="tabular",
        use_label_encoding=use_encoding,
        drop_specific_columns=drop_columns,
    )

    # Model
    x0, _ = next(iter(train_loader))
    input_dim = x0.size(1)
    model = BreastMLPClassifier(
        input_dim=input_dim, num_classes=3, hidden_dims=hidden_dims, dropout=dropout
    ).to(device)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_trainable:,}")

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if custom_weights:
        class_weights = compute_custom_class_weights(
            train_loader, device, malignant_penalty=malignant_penalty
        )
    else:
        class_weights = torch.Tensor([1, 1, 1]).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Train loop
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            train_step(model, train_loader, loss_fn, optimizer, device, writer, epoch)

            acc, val_loss = val_step(model, val_loader, loss_fn, device, writer, epoch)

            pbar.set_description(
                f"Epoch {epoch} | Val Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f}"
            )

    # Save model
    save_model(model, name)

    # Test
    test_acc = test_step(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
