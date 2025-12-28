import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm

from src.data import get_dataloaders
from src.utils import save_model, set_seed, get_class_weights
from src.models import BreastImageClassifier
from src.train_functions import train_step, val_step, test_step


def main():
    epochs: int = 50
    lr: float = 5e-4
    batch_size: int = 32
    dropout: float = 0.5
    info_filename: str = r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    images_folder: str = r"data/BrEaST-Lesions_USG-images_and_masks/"
    backbone: str = "resnet18"
    augment: bool = False
    unfreeze: bool = False
    use_extra_conv: bool = True
    dropout: float = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # empty nohup file
    open("nohup.out", "w").close()

    # Create TensorBoard run
    name = f"Image_model_lr_{lr}_bs_{batch_size}_back_{backbone}_drop_{dropout}_unfreeze_{unfreeze}_extrconv_{use_extra_conv}_augment_{augment}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # Load image data
    train_loader: DataLoader
    val_loader: DataLoader

    train_loader, val_loader, test_loader = get_dataloaders(
        info_file=info_filename,
        images_folder=images_folder,
        batch_size=batch_size,
        augment=augment,
        type="image",
    )

    # Model
    model: BreastImageClassifier = BreastImageClassifier(
        backbone=backbone,
        dropout=dropout,
        num_classes=3,
        unfreeze_last_layer=unfreeze,
        use_extra_conv=use_extra_conv,
    ).to(device)

    num_trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_trainable:,}")

    # Optimizer & loss
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(weight=get_class_weights())
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

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
