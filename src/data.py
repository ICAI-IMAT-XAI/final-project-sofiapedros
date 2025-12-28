import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import random


class DatasetBaseBreast:
    def __init__(
        self,
        info_filename: str,
        images_and_masks_foldername: str,
        use_label_encoding: bool = False,
        drop_specific_columns: bool = False,
    ):
        """
        Args:
        - info_filename (str): path to the Excel file with all the information
        - images_and_masks_foldername (str): path to images and masks
        - use_label_encoding (bool): if True, use label encoding for categorical variables.
                               if False, use one-hot encoding (default behavior)
        - drop_specific_columns (bool): if True, drop Posterior_features, Tissue_composition,
                                  Skin_thickening, Age and Symptons columns
        """

        df: pd.DataFrame = pd.read_excel(
            info_filename, sheet_name="BrEaST-Lesions-USG clinical dat"
        )

        self.df = df.reset_index(drop=True)
        self.images_folder = images_and_masks_foldername
        self.use_label_encoding = use_label_encoding

        # Label mapping
        self.labels = (
            df["Classification"]
            .map({"benign": 0, "malignant": 1, "normal": 2})
            .astype(int)
            .tolist()
        )

        # Save paths
        self.image_paths = [
            os.path.join(images_and_masks_foldername, img)
            for img in df["Image_filename"]
        ]

        self.mask_paths = [
            (
                os.path.join(images_and_masks_foldername, str(file))
                if isinstance(file, str)
                else None
            )
            for file in df["Mask_tumor_filename"]
        ]

        # Tabular
        self.tabular_df = df.copy()

        drop_cols: list = [
            "CaseID",
            "Image_filename",
            "Mask_tumor_filename",
            "Mask_other_filename",
            "Diagnosis",
            "Verification",
            "Interpretation",
            "BIRADS",
            "Classification",
        ]

        # Unnecesary columns according to explanability
        if drop_specific_columns:
            drop_cols.extend(
                [
                    "Posterior_features",
                    "Tissue_composition",
                    "Skin_thickening",
                    "Symptoms",
                    "Age",
                ]
            )

        self.tabular_df = self.tabular_df.drop(columns=drop_cols, errors="ignore")

        # Clean dataset
        # Pass Age from str to int
        for col in self.tabular_df.columns:
            if col.lower() in ["age", "age_years"]:
                self.tabular_df[col] = self.tabular_df[col].replace(
                    ["not available", "Not available", "NOT AVAILABLE", "N/A", "n/a"],
                    np.nan,
                )
                self.tabular_df[col] = pd.to_numeric(
                    self.tabular_df[col], errors="coerce"
                )

        # Convert yes/no columns to binary
        def map_yes_no_na(col):
            if set(col.dropna().unique()).issubset({"yes", "no", "not applicable"}):
                return col.map({"yes": 1, "no": 0, "not applicable": 2}).astype(
                    np.float32
                )
            return col

        self.tabular_df = self.tabular_df.apply(map_yes_no_na)

        categorical_cols = self.tabular_df.select_dtypes(include="object").columns

        # Use label encoding
        if use_label_encoding:
            self.label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                self.tabular_df[col] = self.tabular_df[col].astype(str)
                non_null_mask = (self.tabular_df[col] != "nan") & (
                    self.tabular_df[col].notna()
                )
                if non_null_mask.sum() > 0:
                    self.tabular_df.loc[non_null_mask, col] = le.fit_transform(
                        self.tabular_df.loc[non_null_mask, col]
                    )
                    self.label_encoders[col] = le
                self.tabular_df[col] = (
                    pd.to_numeric(self.tabular_df[col], errors="coerce")
                    .fillna(-1)
                    .astype(np.float32)
                )
        else:
            self.tabular_df = pd.get_dummies(self.tabular_df, columns=categorical_cols)

        # Ensure everything is numeric
        for col in self.tabular_df.columns:
            self.tabular_df[col] = (
                pd.to_numeric(self.tabular_df[col], errors="coerce")
                .fillna(0)
                .astype(np.float32)
            )

        # Normalize
        binary_cols = [
            col
            for col in self.tabular_df.columns
            if set(self.tabular_df[col].unique()).issubset({0.0, 1.0})
        ]
        numeric_cols = self.tabular_df.columns.difference(binary_cols)

        if len(numeric_cols) > 0:
            self.tabular_df[numeric_cols] = (
                self.tabular_df[numeric_cols] - self.tabular_df[numeric_cols].mean()
            ) / (self.tabular_df[numeric_cols].std() + 1e-6)

    def __len__(self) -> int:
        """
        Returns len of the dataset
        """
        return len(self.df)

    def load_image(self, idx: int) -> Image:
        """
        Load the corresponding image
        Args:
        - idx (int): index of the image to load
        Returns:
        - Image
        """
        img_path = self.image_paths[idx]
        return Image.open(img_path).convert("RGB")

    def load_mask(self, idx) -> Image:
        """
        Load the corresponding mask
        Args:
        - idx (int): index of the mask to load
        Returns:
        - Image
        """
        mask_path = self.mask_paths[idx]
        if mask_path is None or not os.path.exists(mask_path):
            return None
        return Image.open(mask_path).convert("L")  # grayscale

    def load_tabular(self, idx: int) -> torch.Tensor:
        """
        Load the corresponding tabular data
        Args:
        - idx (int): index of the mask to load
        Returns:
        - torch.Tensor
        """
        row = self.tabular_df.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float32)

    def load_label(self, idx: int):
        """
        Load the corresponding label
        Args:
        - idx (int): index of the label to load
        Returns:
        - torch.Tensor
        """
        return torch.tensor(self.labels[idx], dtype=torch.long)


class DatasetImagenClasificacion(Dataset):
    def __init__(self, base_dataset: DatasetBaseBreast) -> None:
        """
        Dataset for image classification
        Args:
        - baseDataset (DatasetBaseBreast): complete base Dataset
        """
        self.base = base_dataset

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        """
        Return len of the dataset
        """
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return the corresponding image and label
        Args:
        - idx (int): index of the pair to load
        Returns:
        - tuple with (image, label)
        """
        image = self.base.load_image(idx)
        label = self.base.load_label(idx)
        image = self.transform(image)
        return image, label


class DatasetSegmentacion(Dataset):
    def __init__(self, base_dataset: DatasetBaseBreast):
        """
        Dataset for image classification
        Args:
        - baseDataset (DatasetBaseBreast): complete base Dataset
        """
        self.base = base_dataset

        self.transform_img = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.transform_mask = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        """
        Return len of the dataset
        """
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return the corresponding image, mask and label
        If there is no mask, the  mask is all zeros
        Args:
        - idx (int): index of the pair to load
        Returns:
        - tuple with (image, mask)
        """
        image = self.base.load_image(idx)
        mask = self.base.load_mask(idx)

        image_transformed = self.transform_img(image)

        if mask is None:
            mask_transformed = torch.zeros((1, 224, 224))
        else:
            mask_transformed = self.transform_mask(mask)

        return image_transformed, mask_transformed


class DatasetTabular(Dataset):
    def __init__(self, base_dataset: DatasetBaseBreast) -> None:
        """
        Dataset for classification
        Args:
        - baseDataset (DatasetBaseBreast): complete base Dataset
        """
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return the corresponding data and label
        Args:
        - idx (int): index of the pair to load
        Returns:
        - tuple with (data, label)
        """
        x = self.base.load_tabular(idx)
        y = self.base.load_label(idx)
        return x, y


class ClassAwareAugmentedDataset(Dataset):
    """
    Dataset con oversampling de clases minoritarias mediante augmentación.
    """

    def __init__(
        self, info_filename, images_and_masks_foldername, target_size=(224, 224)
    ):
        breast_dataset = pd.read_excel(
            info_filename, sheet_name="BrEaST-Lesions-USG clinical dat"
        )
        self.images = [
            os.path.join(images_and_masks_foldername, img)
            for img in breast_dataset["Image_filename"]
        ]
        self.labels = (
            breast_dataset["Classification"]
            .map({"benign": 0, "malignant": 1, "normal": 2})
            .astype(int)
            .tolist()
        )

        self.target_size = target_size

        # Base transform
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Transform de augmentación
        self.aug_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )

        # Calcular cuántas veces oversamplear cada clase
        class_counts = np.bincount(self.labels, minlength=3)
        max_count = class_counts.max()

        self.oversampled_images = []
        self.oversampled_labels = []

        for c in range(3):
            idxs = [i for i, l in enumerate(self.labels) if l == c]
            n_to_add = max_count - len(idxs)
            self.oversampled_images.extend([self.images[i] for i in idxs])
            self.oversampled_labels.extend([c for _ in idxs])

            # Oversampling con augmentación
            for _ in range(n_to_add):
                i = random.choice(idxs)
                self.oversampled_images.append(self.images[i])
                self.oversampled_labels.append(c)

    def __len__(self):
        return len(self.oversampled_images)

    def __getitem__(self, idx):
        image = Image.open(self.oversampled_images[idx]).convert("RGB")
        label = self.oversampled_labels[idx]

        # Si la imagen es duplicada por oversampling, aplicar augmentación
        if idx >= len(self.labels):
            image = self.aug_transform(image)

        image = self.base_transform(image)
        return image, label


def get_dataloaders(
    info_file: str,
    images_folder: str,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    seed: int = 42,
    type: str = "tabular",
    use_label_encoding: bool = False,
    drop_specific_columns: bool = False,
    augment: bool = False,
) -> tuple:
    """
    Get the dataloaders for the desired model
    Args:
    - info_filename (str): path to the Excel file with all the information
    - images_folder (str): path to images and masks
    - batch_size (int): batch size
    - split_ratio (float): split ratio between train and val
    - seed (int): seed
    - type (str): type of dataset to load (image, tabular, segmentation)
    - use_label_encoding (bool): if True, use label encoding for categorical variables
    - drop_specific_columns (bool): if True, drop Posterior_features, Tissue_composition,
                                    Skin_thickening, Age and Symptons columns
    Returns:
    - tuple with train_dataloader, val_dataloader, test_dataloader
    """

    base_dataset: DatasetBaseBreast = DatasetBaseBreast(
        info_filename=info_file,
        images_and_masks_foldername=images_folder,
        use_label_encoding=use_label_encoding,
        drop_specific_columns=drop_specific_columns,
    )

    # Generator with fixed seed
    generator = torch.Generator().manual_seed(seed)

    if type == "image":
        if augment:
            dataset = ClassAwareAugmentedDataset(
                info_filename=info_file, images_and_masks_foldername=images_folder
            )
        else:
            dataset = DatasetImagenClasificacion(base_dataset)
    elif type == "tabular":
        dataset = DatasetTabular(base_dataset)
    elif type == "segmentation":
        dataset = DatasetSegmentacion(base_dataset)
    else:
        raise ValueError(
            "Not a valid value of type. Possible values are: 'image', 'tabular' and 'segmentation'"
        )

    # Split between train, val and test
    n_total = len(dataset)
    n_train = int(n_total * split_ratio)
    n_val = int((n_total - n_train) / 2)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    # Get dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
