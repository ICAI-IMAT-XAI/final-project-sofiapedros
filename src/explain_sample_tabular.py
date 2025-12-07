import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

import shap
import os

from src.utils import set_seed, load_model
from src.data import get_dataloaders
from src.data import DatasetBaseBreast
from src.explain_tabular import ModelWrapper, get_feature_names

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs("./results/explainability_s_tabular", exist_ok=True)


def explain_single_sample(
    model_wrapper: ModelWrapper,
    X_train: torch.Tensor,
    sample: torch.Tensor,
    true_label: int,
    feature_names: list,
    sample_idx: int = 0,
    model_name: str = "model",
    class_names: list = ["Benign", "Malignant", "Normal"],
) -> dict:
    """
    Complete explanation for a single sample including prediction,
    SHAP values, and feature contributions.
    Args:
    - model_wrapper (ModelWrapper)
    - X_train (torch.Tensor)
    - sample (torch.Tensor)
    - true_label (int)
    - feature_names (list)
    - sample_idx (int)
    - model_name (str)
    - class_names (list)
    """

    # Ensure correct shape
    if len(sample.shape) == 1:
        sample = sample.unsqueeze(0)

    # Convert tensors to numpy
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    sample_np = sample.cpu().numpy() if isinstance(sample, torch.Tensor) else sample

    # Prediction
    pred_probs = model_wrapper.predict_proba(sample)
    pred_class = np.argmax(pred_probs[0])
    confidence = pred_probs[0][pred_class]

    # SHAP explainer
    background = shap.sample(X_train_np, min(100, len(X_train_np)))
    explainer = shap.KernelExplainer(model_wrapper.predict_proba, background)

    # SHAP values
    shap_values_raw = explainer.shap_values(sample_np)

    # Normalize SHAP format
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray):
        if len(shap_values_raw.shape) == 3:
            shap_values = [
                shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])
            ]
        else:
            shap_values = [shap_values_raw]

    # Generate plots
    plot_single_sample_prediction(
        pred_probs[0], true_label, class_names, sample_idx, model_name
    )

    plot_single_sample_feature_values(
        sample_np[0],
        feature_names,
        shap_values,
        pred_class,
        sample_idx,
        model_name,
        top_n=15,
    )

    return {
        "sample_idx": sample_idx,
        "true_label": true_label,
        "predicted_label": pred_class,
        "confidence": confidence,
        "probabilities": pred_probs[0],
        "shap_values": shap_values,
        "feature_values": sample_np[0],
    }


def plot_single_sample_prediction(
    probabilities: np.ndarray,
    true_label: int,
    class_names: list,
    sample_idx: int,
    model_name: str,
) -> None:
    """
    Plot prpbabilities assigned for each class of a single sample
    as a bar plot
    Args:
    - probabilities (np.ndarray)
    - true_label (int)
    - class_names (list)
    - sample_idx (int)
    - model_name (str)

    """

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "#2ecc71" if i == true_label else "#3498db" for i in range(len(class_names))
    ]
    bars = ax.bar(
        class_names, probabilities * 100, color=colors, alpha=0.7, edgecolor="black"
    )

    pred_class = np.argmax(probabilities)
    bars[pred_class].set_edgecolor("red")
    bars[pred_class].set_linewidth(3)

    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{prob*100:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Probability (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Prediction for Sample {sample_idx}\n(True: {class_names[true_label]}, Predicted: {class_names[pred_class]})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"results/explainability_s_tabular/{model_name}_sample_{sample_idx}_prediction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_single_sample_feature_values(
    sample_values: np.ndarray,
    feature_names: list,
    shap_values: list,
    pred_class: int,
    sample_idx: int,
    model_name: str,
    top_n: int = 15,
) -> None:
    """
    Plot SHAP values for each feature of a single sample
    Args:
    - sample_values (np.ndarray)
    - feature_names (list)
    - shap_values (list)
    - pred_class (int)
    - sample_idx (int)
    - model_name (str)
    - top_n (int): top n features to plot
    """

    shap_vals = shap_values[pred_class][0]

    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))

    top_features = [feature_names[i] for i in top_indices]
    top_values = sample_values[top_indices]
    top_shap = shap_vals[top_indices]

    colors = ["#ff4444" if s > 0 else "#4444ff" for s in top_shap]

    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_values, color=colors, alpha=0.6, edgecolor="black")

    for bar, shap_val in zip(bars, top_shap):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" SHAP: {shap_val:+.3f}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Value (normalized)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Top {top_n} Feature Values for Sample {sample_idx}",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Patch(
            facecolor="#ff4444", alpha=0.6, label="Positive SHAP (increases prediction)"
        ),
        Patch(
            facecolor="#4444ff", alpha=0.6, label="Negative SHAP (decreases prediction)"
        ),
    ]
    ax.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    plt.savefig(
        f"results/explainability_s_tabular/{model_name}_sample_{sample_idx}_feature_values.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def compare_samples(
    model_wrapper: ModelWrapper,
    X_train: torch.Tensor,
    samples: list,
    true_labels: list,
    feature_names: list,
    sample_indices: list,
    model_name: str = "model",
    class_names: list = ["Benign", "Malignant", "Normal"],
) -> None:
    """
    Compare explanations for multiple samples.
    Plots multiple bar plots with the probabilities assigned to
    each class for each sample
    Args:
    - modelWrapper (modelWrapper)
    - X_train (torch.Tensor)
    - samples (list)
    - true_labels (list)
    - feature_names (list)
    - sample_indices (list)
    - model_name (str)
    - class_names (list)
    """

    results = []
    for sample, label, idx in zip(samples, true_labels, sample_indices):
        result = explain_single_sample(
            model_wrapper,
            X_train,
            sample,
            label,
            feature_names,
            idx,
            model_name,
            class_names,
        )
        results.append(result)

    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 6))
    if len(samples) == 1:
        axes = [axes]

    for ax, result, idx in zip(axes, results, sample_indices):
        probs = result["probabilities"]
        true_label = result["true_label"]
        pred_label = result["predicted_label"]

        colors = [
            "#2ecc71" if i == true_label else "#3498db" for i in range(len(class_names))
        ]
        bars = ax.bar(
            class_names, probs * 100, color=colors, alpha=0.7, edgecolor="black"
        )
        bars[pred_label].set_edgecolor("red")
        bars[pred_label].set_linewidth(3)

        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{prob*100:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel("Probability (%)", fontweight="bold")
        ax.set_title(
            f"Sample {idx}\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            fontweight="bold",
        )
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Sample Predictions Comparison", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    plt.savefig(
        f"results/explainability_s_tabular/{model_name}_samples_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
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

    sample_idx = 0
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]

    # Explain
    _ = explain_single_sample(
        model_wrapper,
        X_train,
        X_sample,
        y_sample.item(),
        feature_names,
        sample_idx=sample_idx,
        model_name=model_name,
    )

    # Para comparar varias muestras:
    compare_samples(
        model_wrapper,
        X_train,
        [X_test[0], X_test[1], X_test[3]],
        [y_test[0].item(), y_test[1].item(), y_test[3].item()],
        feature_names,
        [0, 1, 3],
        model_name,
    )


if __name__ == "__main__":
    main()
