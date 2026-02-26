from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute log loss and Brier score for calibrated probability models."""
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y_true, clipped)),
        "brier_score": float(brier_score_loss(y_true, clipped)),
    }


def save_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    bins: int = 12,
) -> Path:
    """Save a calibration curve image for model diagnostics."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
