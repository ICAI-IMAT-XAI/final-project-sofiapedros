import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import shap
import os

from src.utils import set_seed, load_model
from src.data import get_dataloaders, DatasetBaseBreast
from src.explain_tabular import ModelWrapper


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs("./results/xai_evaluation", exist_ok=True)


def compute_shap_values(
    model_wrapper: ModelWrapper,
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_samples: int = 100,
) -> tuple:
    """
    Compute SHAP values for evaluation.

    Args:
    - model_wrapper: Model wrapper with predict_proba method
    - X_train (np.ndarray): Training data
    - X_test (np.ndarray): Test data
    - max_samples (int): Maximum number of samples to compute SHAP values for

    Returns:
    - Tuple[List[np.ndarray], np.ndarray]: SHAP values and sampled data
    """
    background = shap.sample(X_train, min(100, len(X_train)))
    X_sample = X_test[:max_samples]

    explainer = shap.KernelExplainer(model_wrapper.predict_proba, background)
    shap_values_raw = explainer.shap_values(X_sample)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray):
        if len(shap_values_raw.shape) == 3:
            shap_values = [
                shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])
            ]
        else:
            shap_values = [shap_values_raw]

    return shap_values, X_sample


def feature_removal_test(
    model_wrapper: ModelWrapper,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    model_name: str,
    n_samples: int = 10,
    top_k: int = 5,
) -> dict:
    """
    Feature removal test: removing the most important features should
    significantly reduce model performance.

    Args:
    - model_wrapper: Model wrapper with predict and predict_proba methods
    - X_train (np.ndarray): Training data
    - X_test (np.ndarray): Test data
    - y_test (np.ndarray): Test labels
    - feature_names (List[str]): List of feature names
    - model_name (str): Name of the model for saving plots
    - n_samples (int): Number of samples to test
    - top_k (int): Number of top features to remove

    Returns:
    - Dict[str, Any]: Dictionary with test results
    """
    # Compute SHAP values
    shap_values, X_sample = compute_shap_values(
        model_wrapper, X_train, X_test, max_samples=n_samples
    )

    # Get feature importance (average across samples and classes)
    feature_importance = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
    )
    top_features = np.argsort(feature_importance)[-top_k:][::-1]

    # Original accuracy
    y_sample = y_test[:n_samples]
    original_preds = model_wrapper.predict(X_sample)
    original_probs = model_wrapper.predict_proba(X_sample)
    original_acc = accuracy_score(y_sample, original_preds)
    original_conf = np.mean(
        [original_probs[i, original_preds[i]] for i in range(len(original_preds))]
    )

    print(f"\nOriginal performance:")
    print(f"Accuracy: {original_acc*100:.2f}%")
    print(f"Mean confidence: {original_conf*100:.2f}%")

    # Remove top features (assign mean value)
    X_perturbed = X_sample.copy()
    feature_means = X_train.mean(axis=0)
    X_perturbed[:, top_features] = feature_means[top_features]

    perturbed_preds = model_wrapper.predict(X_perturbed)
    perturbed_probs = model_wrapper.predict_proba(X_perturbed)
    perturbed_acc = accuracy_score(y_sample, perturbed_preds)
    perturbed_conf = np.mean(
        [perturbed_probs[i, perturbed_preds[i]] for i in range(len(perturbed_preds))]
    )

    print(f"\nPerformance after removing top {top_k} features:")
    print(f"Accuracy: {perturbed_acc*100:.2f}%")
    print(f"Mean confidence: {perturbed_conf*100:.2f}%")

    acc_drop = (original_acc - perturbed_acc) * 100
    conf_drop = (original_conf - perturbed_conf) * 100

    print("\n")
    print(f"Accuracy drop: {acc_drop:.2f}%")
    print(f"Confidence drop: {conf_drop:.2f}%")

    # Test with random features
    random_features = np.random.choice(len(feature_names), top_k, replace=False)
    X_random = X_sample.copy()
    X_random[:, random_features] = feature_means[random_features]

    random_preds = model_wrapper.predict(X_random)
    random_acc = accuracy_score(y_sample, random_preds)
    random_acc_drop = (original_acc - random_acc) * 100

    print(f"\nComparison with random feature removal:")
    print(f"  Accuracy drop with random features: {random_acc_drop:.2f}%")

    # Visualize
    _plot_feature_removal_test(
        original_acc, perturbed_acc, random_acc, top_k, model_name
    )

    return {
        "original_accuracy": original_acc,
        "perturbed_accuracy": perturbed_acc,
        "random_accuracy": random_acc,
        "accuracy_drop": acc_drop,
        "confidence_drop": conf_drop,
        "top_features": [feature_names[i] for i in top_features],
    }


def monotonicity_test(
    model_wrapper: ModelWrapper,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_samples: int = 20,
    n_steps: int = 10,
) -> dict:
    """
    Monotonicity test: when removing features in order of importance,
    performance should decrease monotonically.

    Args:
    - model_wrapper: Model wrapper with predict method
    - X_train (np.ndarray): Training data
    - X_test (np.ndarray): Test data
    - y_test (np.ndarray): Test labels
    - model_name (str): Name of the model for saving plots
    -- n_samples (int): Number of samples to test
    - n_steps (int): Number of removal steps

    Returns:
    - Dict[str, Any]: Dictionary with test results
    """

    # Compute SHAP values
    shap_values, X_sample = compute_shap_values(
        model_wrapper, X_train, X_test, max_samples=n_samples
    )
    y_sample = y_test[:n_samples]

    # Get feature importance
    feature_importance = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
    )
    sorted_features = np.argsort(feature_importance)[::-1]  # Most to least important

    # Original accuracy
    original_acc = accuracy_score(y_sample, model_wrapper.predict(X_sample))

    # Progressively remove features
    accuracies = [original_acc]
    n_features_removed = [0]

    step_size = max(1, len(sorted_features) // n_steps)
    feature_means = X_train.mean(axis=0)

    for step in range(1, n_steps + 1):
        # Remove next batch of features
        end_idx = min(step * step_size, len(sorted_features))
        features_to_remove = sorted_features[:end_idx]

        X_current = X_sample.copy()
        X_current[:, features_to_remove] = feature_means[features_to_remove]

        acc = accuracy_score(y_sample, model_wrapper.predict(X_current))
        accuracies.append(acc)
        n_features_removed.append(len(features_to_remove))

        print(f"Removed {len(features_to_remove)} features: Accuracy = {acc*100:.2f}%")

    # Verify monotonicity (should generally decrease)
    monotonic_decreases = sum(
        [accuracies[i] >= accuracies[i + 1] for i in range(len(accuracies) - 1)]
    )
    monotonicity_ratio = monotonic_decreases / (len(accuracies) - 1)

    print(f"\nResults:")
    print(f"  Monotonic decreases: {monotonic_decreases}/{len(accuracies)-1}")
    print(f"  Monotonicity ratio: {monotonicity_ratio:.2f}")

    # Visualize
    _plot_monotonicity_test(n_features_removed, accuracies, model_name)

    return {
        "monotonicity_ratio": monotonicity_ratio,
        "accuracies": accuracies,
        "n_features_removed": n_features_removed,
    }


def noise_sensitivity_test(
    model_wrapper: ModelWrapper,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    model_name: str,
    n_samples: int = 20,
    top_k: int = 5,
    noise_levels: list = [0.1, 0.5, 1.0, 2.0],
) -> dict:
    """
    Noise sensitivity test: adding noise to the most important features
    should significantly change model predictions.

    Args:
    - model_wrapper: Model wrapper with predict and predict_proba methods
    - X_train (np.ndarray): Training data
    - X_test (np.ndarray): Test data
    - y_test (np.ndarray): Test labels
    - feature_names (List[str]): List of feature names
    - model_name (str): Name of the model for saving plots
    - n_samples (int): Number of samples to test
    - top_k (int): Number of top features to add noise to
    - noise_levels (List[float]): List of noise levels to test

    Returns:
    - Dict[str, Any]: Dictionary with test results
    """
    # Compute SHAP values
    shap_values, X_sample = compute_shap_values(
        model_wrapper, X_train, X_test, max_samples=n_samples
    )
    y_sample = y_test[:n_samples]

    # Get feature importance
    feature_importance = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
    )

    # Adjust top_k if necessary
    actual_top_k = min(top_k, len(feature_names))
    top_features = np.argsort(feature_importance)[-actual_top_k:][::-1]

    # Also select random features for comparison
    available_features = [i for i in range(len(feature_names)) if i not in top_features]
    k_random = min(actual_top_k, len(available_features))

    if len(available_features) == 0:
        # Use same features as important ones to avoid breaking the code
        random_features = top_features[:k_random]
    else:
        random_features = np.random.choice(
            available_features,
            k_random,
            replace=False,
        )

    # Original predictions
    original_preds = model_wrapper.predict(X_sample)
    original_probs = model_wrapper.predict_proba(X_sample)
    original_acc = accuracy_score(y_sample, original_preds)

    print(f"\nOriginal accuracy: {original_acc*100:.2f}%")

    results_important = []
    results_random = []

    # Calculate feature statistics to scale noise appropriately
    feature_stds = X_train.std(axis=0)

    for noise_level in noise_levels:
        print(f"\n--- Noise level: {noise_level} ---")

        # Noise on important features
        X_noisy_important = X_sample.copy()
        for feat_idx in top_features:
            noise = np.random.normal(
                0, noise_level * feature_stds[feat_idx], X_noisy_important.shape[0]
            )
            X_noisy_important[:, feat_idx] += noise

        noisy_preds_important = model_wrapper.predict(X_noisy_important)
        noisy_probs_important = model_wrapper.predict_proba(X_noisy_important)
        noisy_acc_important = accuracy_score(y_sample, noisy_preds_important)

        # Changes in predictions
        pred_changes_important = np.sum(original_preds != noisy_preds_important)

        # Changes in probabilities
        prob_changes_important = np.mean(np.abs(original_probs - noisy_probs_important))

        results_important.append(
            {
                "noise_level": noise_level,
                "accuracy": noisy_acc_important,
                "prediction_changes": pred_changes_important,
                "prob_change": prob_changes_important,
            }
        )

        print(f"Important features with noise {noise_level}:")
        print(f"Accuracy: {noisy_acc_important*100:.2f}%")
        print(f"Predictions changed: {pred_changes_important}/{n_samples}")
        print(f"Mean probability change: {prob_changes_important:.4f}")

        # Noise on random features (for comparison)
        X_noisy_random = X_sample.copy()
        for feat_idx in random_features:
            noise = np.random.normal(
                0, noise_level * feature_stds[feat_idx], X_noisy_random.shape[0]
            )
            X_noisy_random[:, feat_idx] += noise

        noisy_preds_random = model_wrapper.predict(X_noisy_random)
        noisy_probs_random = model_wrapper.predict_proba(X_noisy_random)
        noisy_acc_random = accuracy_score(y_sample, noisy_preds_random)

        pred_changes_random = np.sum(original_preds != noisy_preds_random)
        prob_changes_random = np.mean(np.abs(original_probs - noisy_probs_random))

        results_random.append(
            {
                "noise_level": noise_level,
                "accuracy": noisy_acc_random,
                "prediction_changes": pred_changes_random,
                "prob_change": prob_changes_random,
            }
        )

        print(f"\nRandom features with noise {noise_level}:")
        print(f"Accuracy: {noisy_acc_random*100:.2f}%")
        print(f"Predictions changed: {pred_changes_random}/{n_samples}")
        print(f"Mean probability change: {prob_changes_random:.4f}")

    # Visualize
    _plot_noise_test(
        results_important, results_random, noise_levels, n_samples, model_name
    )

    return {
        "original_accuracy": original_acc,
        "important_features_results": results_important,
        "random_features_results": results_random,
        "top_features": [feature_names[i] for i in top_features],
    }


def _plot_feature_removal_test(
    original_acc: float,
    perturbed_acc: float,
    random_acc: float,
    top_k: int,
    model_name: str,
) -> None:
    """
    Visualize feature removal test results.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Original", f"Top {top_k}\nRemoved", f"Random {top_k}\nRemoved"]
    accuracies = [original_acc * 100, perturbed_acc * 100, random_acc * 100]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    bars = ax.bar(
        categories,
        accuracies,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        width=0.6,
    )

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Feature Removal Test (Faithfulness)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"results/xai_evaluation/{model_name}_feature_removal_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _plot_monotonicity_test(
    n_features_removed: list, accuracies: list, model_name: str
) -> None:
    """
    Visualize monotonicity test results.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        n_features_removed,
        [acc * 100 for acc in accuracies],
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )

    ax.set_xlabel("Number of Features Removed", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Monotonicity Test: Accuracy vs Features Removed",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"results/xai_evaluation/{model_name}_monotonicity_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _plot_noise_test(
    results_important: list,
    results_random: list,
    noise_levels: list,
    n_samples: int,
    model_name: str,
) -> None:
    """
    Visualize noise sensitivity test results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Accuracy vs Noise Level
    ax1 = axes[0]
    acc_important = [r["accuracy"] * 100 for r in results_important]
    acc_random = [r["accuracy"] * 100 for r in results_random]

    ax1.plot(
        noise_levels,
        acc_important,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Important features",
        color="#e74c3c",
    )
    ax1.plot(
        noise_levels,
        acc_random,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Random features",
        color="#3498db",
    )

    ax1.set_xlabel("Noise Level", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Accuracy vs Noise Level", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Prediction Changes
    ax2 = axes[1]
    pred_changes_important = [r["prediction_changes"] for r in results_important]
    pred_changes_random = [r["prediction_changes"] for r in results_random]

    ax2.plot(
        noise_levels,
        pred_changes_important,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Important features",
        color="#e74c3c",
    )
    ax2.plot(
        noise_levels,
        pred_changes_random,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Random features",
        color="#3498db",
    )

    ax2.set_xlabel("Noise Level", fontsize=11, fontweight="bold")
    ax2.set_ylabel(
        f"Predictions Changed (out of {n_samples})", fontsize=11, fontweight="bold"
    )
    ax2.set_title("Prediction Changes", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Probability Changes
    ax3 = axes[2]
    prob_changes_important = [r["prob_change"] for r in results_important]
    prob_changes_random = [r["prob_change"] for r in results_random]

    ax3.plot(
        noise_levels,
        prob_changes_important,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Important features",
        color="#e74c3c",
    )
    ax3.plot(
        noise_levels,
        prob_changes_random,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Random features",
        color="#3498db",
    )

    ax3.set_xlabel("Noise Level", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean Probability Change", fontsize=11, fontweight="bold")
    ax3.set_title("Probability Changes", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.suptitle(
        "Noise Sensitivity Test on Important Features",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        f"results/xai_evaluation/{model_name}_noise_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def run_all_xai_tests(
    model_wrapper: ModelWrapper,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    model_name: str,
    max_samples: int = 50,
) -> dict:
    """
    Run all XAI evaluation tests.

    Args:
    - model_wrapper: Model wrapper with predict and predict_proba methods
    - X_train (np.ndarray): Training data
    - X_test (np.ndarray): Test data
    - y_test (np.ndarray): Test labels
    - feature_names (List[str]): List of feature names
    - model_name (str): Name of the model for saving plots
    - max_samples (int): Maximum number of samples for testing

    Returns:
    - Dict[str, Any]: Dictionary containing all test results
    """

    results = {}

    # Run tests
    results["feature_removal"] = feature_removal_test(
        model_wrapper,
        X_train,
        X_test,
        y_test,
        feature_names,
        model_name,
        n_samples=max_samples,
        top_k=5,
    )
    results["monotonicity"] = monotonicity_test(
        model_wrapper,
        X_train,
        X_test,
        y_test,
        model_name,
        n_samples=max_samples,
        n_steps=10,
    )
    results["noise_sensitivity"] = noise_sensitivity_test(
        model_wrapper,
        X_train,
        X_test,
        y_test,
        feature_names,
        model_name,
        n_samples=max_samples,
        top_k=5,
        noise_levels=[0.1, 0.5, 1.0, 2.0],
    )

    return results


def main() -> None:
    """
    Main function to run XAI evaluation.
    """
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
    train_loader, _, test_loader = get_dataloaders(
        info_filename,
        images_and_masks_foldername,
        batch_size,
        seed=seed,
        type="tabular",
        use_label_encoding=True,
        drop_specific_columns=True,
    )

    model = load_model(model_name)
    model_wrapper = ModelWrapper(model, device)

    # Get feature names
    base_dataset = DatasetBaseBreast(
        info_filename,
        images_and_masks_foldername,
        use_label_encoding=True,
        drop_specific_columns=True,
    )
    feature_names = list(base_dataset.tabular_df.columns)

    # Prepare data
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

    # Convert to numpy
    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Run all tests
    _ = run_all_xai_tests(
        model_wrapper,
        X_train_np,
        X_test_np,
        y_test_np,
        feature_names,
        model_name,
        max_samples=50,
    )


if __name__ == "__main__":
    main()
