import torch
import torch.nn as nn
from torch.jit import RecursiveScriptModule
import numpy as np
import os
import random
import pandas as pd

N_CLASSES: int = 3
info_filename: str = "data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed number to fix radomness
    """

    # Set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # Set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
    - model (nn.Module): pytorch model.
    - name (str): name of the model (without the extension).
    """

    # Create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # Save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
    - name (str): name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    try:
        model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")
    except ValueError:
        model: RecursiveScriptModule = torch.jit.load(f"../models/{name}.pt")
    return model


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
    - predictions (torch.Tensor): predictions tensor. Dimensions:
            [batch, num classes] or [batch].
    - targets (torch.Tensor): targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
        the accuracy in a tensor of a single element.
    """

    # Get the value of the predictions -
    # the index associated with the highest probability
    real_predictions = torch.argmax(predictions, dim=1)

    # Compute accuracy as well-classifies/total
    accuracy_value = torch.mean((real_predictions == targets).float()).item()
    return accuracy_value


def dice_coeff(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate dice coefficient.
    Args:
    - pred (torch.Tensor): predictions
    - target (torch.Tensor): target
    - eps (float): epsilon
    Returns:
    - torch.tensor dice coefficient
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    return torch.mean((2 * intersection + eps) / (union + eps))


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        BCE + DICE
        """
        bce = self.bce(pred.float(), target.float())

        dice = 1 - dice_coeff(pred, target)
        return bce + dice
