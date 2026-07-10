# -*- coding: utf-8 -*-
"""Feature engineering görselleştirmeleri."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .paths import fe_figures


def plot_feature_overview(feat_30min: pd.DataFrame, section: str = "overview") -> str:
    """Temel feature dağılımları."""
    out = fe_figures(section)
    cols = ["kayip_kazanc", "kk_oran", "doluluk_oran", "tx_ue1t_fark"]
    cols = [c for c in cols if c in feat_30min.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = feat_30min[col].dropna()
        if col == "kk_oran":
            data = data.clip(-0.2, 0.2)
        ax.hist(data, bins=50, edgecolor="k", alpha=0.7)
        ax.set_title(col)
    plt.tight_layout()
    path = out / "feature_distributions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_null_heatmap(feat: pd.DataFrame, section: str = "nulls") -> str:
    """Feature tablosu null oranı."""
    out = fe_figures(section)
    null_pct = (feat.isna().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0].head(20)
    fig, ax = plt.subplots(figsize=(8, max(3, len(null_pct) * 0.3)))
    null_pct.plot(kind="barh", ax=ax, color="coral", edgecolor="k")
    ax.set_xlabel("Null (%)")
    ax.set_title("Feature null oranları (top 20)")
    plt.tight_layout()
    path = out / "null_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_signal_counts(feat_30min: pd.DataFrame, section: str = "signals") -> str:
    """Kural tabanlı sinyal sayıları."""
    out = fe_figures(section)
    sig_cols = [c for c in feat_30min.columns if c.startswith("sinyal_")]
    counts = feat_30min[sig_cols].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="k")
    ax.set_xlabel("Toplam tetiklenme")
    ax.set_title("Kural tabanlı sinyaller")
    plt.tight_layout()
    path = out / "signal_counts.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)
