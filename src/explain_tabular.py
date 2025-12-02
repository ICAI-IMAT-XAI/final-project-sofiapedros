import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

import shap
import os

from src.utils import set_seed, load_model
from src.data import get_dataloaders
from src.data import DatasetBaseBreast


device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs("./results/explainability_tabular", exist_ok=True)


class ModelWrapper:
    """
    Wrapper to fix compatibility issues with SHAP and sklearn
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def fit(self, X, y):
        """
        Dummy function for SHAP compatibility
        """
        return self

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict model output
        """

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Probability prediction
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def score(self, X, y):
        """
        Function required by skelearn to compute accuracy
        """
        preds = self.predict(X)
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        return accuracy_score(y, preds)


def get_feature_names(base_dataset: Dataset) -> list:
    """
    Get dataset feature names as a list
    Args:
    - base_dataset (Dataset)
    """
    return list(base_dataset.tabular_df.columns)


def compute_permutation_importance(
    model_wrapper: ModelWrapper,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_names: list,
    n_repeats: int = 10,
    model_name: str = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_False",
) -> tuple:
    """
    Computes permutation importance
    Args:
    - model_wrapper (ModelWrapper)
    - X_test (torch.Tensor)
    - y_test (torch.Tenosr)
    - feature_names (list)
    - n_repeats (int)
    - model_name (str)
    Returns:
    - dict with permutation importance
    - sorted indices
    """

    # Convert to numpy if necessary
    if isinstance(X_test, torch.Tensor):
        X_test_np = X_test.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
    else:
        X_test_np = X_test
        y_test_np = y_test

    # Permutation importance with sklearn
    perm_importance = permutation_importance(
        model_wrapper,
        X_test_np,
        y_test_np,
        n_repeats=n_repeats,
        random_state=42,
        scoring="accuracy",
    )

    # Sort by importance
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    # Show 20 features
    print("\nTop features:")
    for i, idx in enumerate(sorted_idx[:20]):
        print(
            f"{i+1}. {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} "
            f"(+/- {perm_importance.importances_std[idx]:.4f})"
        )

    # Visualize
    plot_permutation_importance(
        perm_importance, feature_names, sorted_idx, top_n=30, model_name=model_name
    )

    return perm_importance, sorted_idx


def plot_permutation_importance(
    perm_importance: dict,
    feature_names: list,
    sorted_idx: list,
    top_n: int = 20,
    model_name: str = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_False",
) -> None:
    """
    Visualizes permutation importance
    Args:
    - perm_importance (dict)
    - feature_names (list)
    - sorted_idx (list)
    - n_repeats (int)
    - model_name (str)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top_idx = sorted_idx[:top_n]
    importances = perm_importance.importances_mean[top_idx]
    stds = perm_importance.importances_std[top_idx]
    top_features = [feature_names[i] for i in top_idx]

    # Bar plot
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, importances, xerr=stds, align="center", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (decrease in accuracy)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Features - Permutation Importance", fontsize=13, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"results/explainability_tabular/{model_name}_permutation_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def compute_shap_values(
    model_wrapper: ModelWrapper,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    feature_names: list,
    max_samples: int = 100,
    model_name: str = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_False",
) -> tuple:
    """
    Computes SHAP Values with KernelExplainer
    Args:
    - model_wrapper (ModelWrapper)
    - X_test (torch.Tensor)
    - y_test (torch.Tenosr)
    - feature_names (list)
    - max_samples (int)
    - model_name (str)
    Returns:
    - shap_values, X_test_sample
    """
    # Convert to numpy
    if isinstance(X_train, torch.Tensor):
        X_train_np = X_train.cpu().numpy()
        X_test_np = X_test.cpu().numpy()
    else:
        X_train_np = X_train
        X_test_np = X_test

    # Use one sample of test loader as background
    background = shap.sample(X_train_np, min(100, len(X_train_np)))

    # Use max samples to accelerate
    X_test_sample = X_test_np[:max_samples]

    # Create explainer
    explainer = shap.KernelExplainer(model_wrapper.predict_proba, background)

    # Compute shap values
    shap_values_raw = explainer.shap_values(X_test_sample)

    # Normalize type
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray):
        if len(shap_values_raw.shape) == 3:
            shap_values = [
                shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])
            ]
        else:
            shap_values = [shap_values_raw]
    else:
        raise ValueError(f"Unexpected format for SHAP values: {type(shap_values_raw)}")

    print(f"Shape de X_test_sample: {X_test_sample.shape}")

    # Visualizations
    plot_shap_summary(shap_values, X_test_sample, feature_names, model_name=model_name)
    plot_shap_bar(shap_values, feature_names, model_name=model_name)

    return shap_values, X_test_sample


def plot_shap_summary(
    shap_values: list,
    X_test: np.ndarray,
    feature_names: list,
    model_name: str = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_False",
):
    """
    SHAP summary plot
    Args:
    - shap_values (list)
    - X_test (np.ndarray)
    - feature_names (list)
    - model_name (str)
    """
    class_names = ["Benign", "Malignant", "Normal"]

    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    for class_idx, class_name in enumerate(class_names):
        if class_idx >= len(shap_values):
            print(f"Warning: No shap values for class: {class_name}")
            continue

        plt.figure(figsize=(10, 8))

        # Verify dimensions match
        sv = shap_values[class_idx]

        if sv.shape[1] != X_test.shape[1]:
            print(f"  ERROR: Dimensiones no coinciden. Saltando...")
            plt.close()
            continue

        shap.summary_plot(
            sv, X_test, feature_names=feature_names, show=False, max_display=20
        )
        plt.title(
            f"SHAP Summary Plot - {class_name}", fontsize=13, fontweight="bold", pad=20
        )
        plt.tight_layout()
        plt.savefig(
            f"results/explainability_tabular/{model_name}_shap_summary_{class_name.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_shap_bar(
    shap_values: list,
    feature_names: list,
    top_n: int = 20,
    model_name: str = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_False",
) -> None:
    """
    SHAP bar plot with average importance
    - shap_values (list)
    - X_test (np.ndarray)
    - feature_names (list)
    - model_name (str)
    """
    class_names = ["Benign", "Malignant", "Normal"]

    # Verify array
    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    for class_idx, class_name in enumerate(class_names):
        if class_idx >= len(shap_values):
            print(f"Warning: No shap values for class: {class_name}")
            continue

        plt.figure(figsize=(10, 8))

        sv = shap_values[class_idx]

        # Verify dimensions
        if sv.shape[1] != len(feature_names):
            print(
                f"Advertencia: Dimensiones no coinciden para {class_name}. Saltando..."
            )
            plt.close()
            continue

        # Compute absolute value of mean importance
        mean_abs_shap = np.abs(sv).mean(axis=0)

        # Sort and keep top n
        sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in sorted_idx.tolist()]
        sorted_values = mean_abs_shap[sorted_idx]

        # Plot
        y_pos = np.arange(len(sorted_features))
        plt.barh(y_pos, sorted_values, alpha=0.7, color="steelblue")
        plt.yticks(y_pos, sorted_features, fontsize=9)
        plt.gca().invert_yaxis()
        plt.xlabel("Mean |SHAP value|", fontsize=11)
        plt.title(
            f"Top {top_n} Features - SHAP Importance ({class_name})",
            fontsize=13,
            fontweight="bold",
        )
        plt.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"results/explainability_tabular/{model_name}_shap_bar_{class_name.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def create_comparison_plot(
    perm_importance: dict,
    shap_values: list,
    feature_names: list,
    sorted_idx_perm: list,
    top_n: int = 15,
    model_name: str = "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_False",
):
    """
    Compares permutation importance with SHAP
    Args:
    - perm_importance (dict)
    - shap_values (list)
    - feature_names (list)
    - sorted_idx_perm (list)
    - top_n (int)
    - model_name (str)
    """

    # Verify array
    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    # Mean SHAP for all classes
    shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    # Verificar dimensiones
    if len(shap_importance) != len(feature_names):
        print(f"Warning: Dimensions do not match.")
        print(f"  SHAP: {len(shap_importance)}, Features: {len(feature_names)}")
        return

    # Normalize metrics
    perm_norm = (
        perm_importance.importances_mean / perm_importance.importances_mean.max()
    )
    shap_norm = shap_importance / shap_importance.max()

    # Obtain top features
    top_idx = sorted_idx_perm[:top_n]
    top_features = [feature_names[i] for i in top_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    y_pos = np.arange(len(top_features))

    # Permutation importance
    ax1.barh(y_pos, perm_norm[top_idx], alpha=0.7, color="coral")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Normalized Importance", fontsize=11)
    ax1.set_title("Permutation Importance", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # SHAP importance
    ax2.barh(y_pos, shap_norm[top_idx], alpha=0.7, color="steelblue")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_features, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Normalized Importance", fontsize=11)
    ax2.set_title("SHAP Importance (averaged)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Comparison: Top {top_n} Features", fontsize=14, fontweight="bold", y=1.00
    )
    plt.tight_layout()
    plt.savefig(
        f"results/explainability_tabular/{model_name}_comparison_perm_vs_shap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    # Configuration
    model_name: str = (
        "Tab_model_lr_0.0001_bs_8_hd_(128, 256)_dropout_0.2_encode_True_drop_True_cw_True_1.5"
    )
    batch_size: int = 32
    seed: int = 42
    info_filename: str = r"data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    images_and_masks_foldername: str = r"data/BrEaST-Lesions_USG-images_and_masks/"

    set_seed(seed)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        info_filename,
        images_and_masks_foldername,
        batch_size,
        seed=seed,
        type="tabular",
        use_label_encoding=True,
        drop_specific_columns=True,
    )

    # Load model
    model = load_model(model_name)
    model_wrapper = ModelWrapper(model, device)

    # Obtain feature names
    base_dataset = DatasetBaseBreast(
        info_filename,
        images_and_masks_foldername,
        use_label_encoding=True,
        drop_specific_columns=True,
    )
    feature_names = get_feature_names(base_dataset)
    print(f"Number of features: {len(feature_names)}")

    # Prepare data for SHAP and perm importance
    X_test_list, y_test_list = [], []
    for X, y in test_loader:
        X_test_list.append(X)
        y_test_list.append(y)
    X_test = torch.cat(X_test_list)
    y_test = torch.cat(y_test_list)

    X_train_list = []
    for X, _ in train_loader:
        X_train_list.append(X)
    X_train = torch.cat(X_train_list)

    # Accuracy
    print("\Evaluating model...")
    preds = model_wrapper.predict(X_test)
    acc = accuracy_score(y_test.cpu().numpy(), preds)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # 1. Permutation Importance
    perm_importance, sorted_idx = compute_permutation_importance(
        model_wrapper,
        X_test,
        y_test,
        feature_names,
        n_repeats=10,
        model_name=model_name,
    )

    # 2. SHAP Values
    shap_values, X_test_sample = compute_shap_values(
        model_wrapper,
        X_train,
        X_test,
        feature_names,
        max_samples=100,
        model_name=model_name,
    )

    # 3. Compare
    create_comparison_plot(
        perm_importance, shap_values, feature_names, sorted_idx, model_name=model_name
    )


if __name__ == "__main__":
    main()
