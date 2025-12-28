import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch


class BreastImageClassifier(nn.Module):
    """
    Fast, efficient breast lesion classifier using EfficientNet-B0 (default).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        unfreeze_last_layer: bool = True,
        use_extra_conv: bool = False,
    ):
        """
        Args:
        - backbone (str): backbone. Options: resnet18, resnet50
        - pretrained (bool): use a pretrained backbone or not
        - num_classes (int): number of classes
        - dropout (float): dropout rate
        """
        super().__init__()

        # 1. Load backbone
        if backbone == "resnet18":
            weights = (
                models.resnet.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            net = models.resnet18(weights=weights)
            last_c = 512
        else:
            weights = (
                models.resnet.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            net = models.resnet50(weights=weights)
            last_c = 2048

        # 2. Remove ONLY the FC layer — keep everything else intact
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "layer4") and unfreeze_last_layer:
            for name, child in self.backbone.named_children():
                if name == "layer4":
                    for param in child.parameters():
                        param.requires_grad = True
                    break

        # 3. Optional extra conv block
        self.use_extra_conv = use_extra_conv
        if use_extra_conv:
            self.extra_conv = nn.Sequential(
                nn.Conv2d(last_c, last_c // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_c // 2, last_c // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            final_c = last_c // 4
        else:
            self.extra_conv = nn.Identity()
            final_c = last_c

        # 4. Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(final_c, num_classes))

    def forward(self, x):
        # Backbone intacto
        x = self.backbone(x)

        # Extra conv opcional
        x = self.extra_conv(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BreastSegmentationModel(nn.Module):
    def __init__(self, num_classes=1, train_last_encoder_block=False):
        super().__init__()

        # Crear UNet preentrenado
        self.model = segmentation_models_pytorch.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

        for param in self.model.encoder.parameters():
            param.requires_grad = False

        if train_last_encoder_block:
            for param in self.model.encoder.layer4.parameters():
                param.requires_grad = True

        for param in self.model.decoder.parameters():
            param.requires_grad = True

        for param in self.model.segmentation_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class BreastMLPClassifier(nn.Module):
    """
    Simple, efficient MLP for tabular data
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dims: tuple = (128, 64),
        dropout: float = 0.2,
    ):
        """
        Args:
        - input_dim (int)
        - num_classes (int)
        - hidden_dims (tuple)
        - dropout (float)
        """
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev = h

        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
