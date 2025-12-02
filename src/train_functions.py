import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.utils import accuracy, dice_coeff


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    segmentation: bool = False,
) -> None:
    """
    This function computes the training step and logs metrics to TensorBoard.

    Args:
    - model (nn.Module): pytorch model.
    - train_data (Dataloader): train dataloader.
    - loss (torch.nn.Module): loss function.
    - optimizer (torch.optim.Optimizer): optimizer object.
    - device (torch.device): device of model.
    - writer (SummaryWriter): TensorBoard writer.
    - epoch (int): current epoch number.
    - segmentation (bool)
    """
    model.train()

    losses: list[float] = []
    accuracies: list[float] = []

    for _, (image, target) in enumerate(train_data):

        image = image.to(device)
        if segmentation:
            target = target.to(device).float()
        else:
            target = target.to(device).long()

        outputs = model(image)

        loss_value = loss(outputs, target.long())
        accuracy_value = accuracy(outputs, target)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        accuracies.append(accuracy_value)

    # Log average per epoch
    writer.add_scalar("Train/Average_Loss", np.mean(losses), epoch)
    writer.add_scalar("Train/Average_Accuracy", np.mean(accuracies), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    segmentation: bool = False,
) -> tuple[float, float]:
    """
    Computes the validation step and logs metrics to TensorBoard.

    Args:
    - model (nn.Module): pytorch model.
    - train_data (Dataloader): train dataloader.
    - loss (torch.nn.Module): loss function.
    - device (torch.device): device of model.
    - writer (SummaryWriter): TensorBoard writer.
    - epoch (int): current epoch number.
    - segmentation (bool)
    Returns:
        average accuracy, average loss
    """
    model.eval()

    losses = []
    accuracies = []
    if segmentation:
        dice_values = []

    with torch.no_grad():
        for _, (image, target) in enumerate(val_data):
            image = image.to(device)
            if segmentation:
                target = target.to(device).float()
            else:
                target = target.to(device)

            outputs = model(image)

            loss_value = loss(outputs, target.long())
            if segmentation:
                dice_value = dice_coeff(outputs, target)
                dice_values.append(dice_value.item())

            accuracy_value = accuracy(outputs, target)

            losses.append(loss_value.item())
            accuracies.append(accuracy_value)

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)

    # Log average per epoch
    writer.add_scalar("Val/Average_Loss", avg_loss, epoch)
    writer.add_scalar("Val/Average_Accuracy", avg_acc, epoch)

    if segmentation:
        avg_dice = np.mean(dice_values)
        return avg_acc, avg_loss, avg_dice

    return avg_acc, avg_loss


def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
    segmentation: bool = False,
) -> float:
    """
    Computes the test step. Logging is optional for testing.

    Args:
    - model (nn.Module): pytorch model.
    - test_data (DataLoader): dataloader of test data.
    - device (torch.device): device of model.
    - segmentation (bool): whether it's a segmentation task.

    Returns:
        average accuracy
    """
    model.eval()
    model = model.to(device)
    accuracies = []
    if segmentation:
        dice_values = []

    with torch.no_grad():
        for coordinates, target in test_data:
            coordinates = coordinates.to(device)
            target = target.to(device)

            outputs = model(coordinates)
            accuracy_value = accuracy(outputs, target)
            accuracies.append(accuracy_value)

            if segmentation:
                dice_value = dice_coeff(outputs, target)
                dice_values.append(dice_value.item())

    if segmentation:
        return np.mean(accuracies), np.mean(dice_values)

    return np.mean(accuracies)
