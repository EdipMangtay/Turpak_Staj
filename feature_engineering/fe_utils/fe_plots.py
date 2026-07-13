# -*- coding: utf-8 -*-
"""Feature engineering görselleştirmeleri — notebook içinde inline (plt.show)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _show():
    """Grafiği notebook hücresinde göster (Cursor/VS Code + nbconvert uyumlu)."""
    fig = plt.gcf()
    plt.tight_layout()
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
        else:
            plt.show()
    except ImportError:
        plt.show()
    plt.close(fig)


def show_fig():
    """Notebook hücrelerinde manuel grafik göstermek için."""
    _show()


def setup_fe_style():
    plt.rcParams.update({
        "figure.figsize": (12, 4),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "figure.dpi": 100,
    })
    sns.set_theme(style="whitegrid", palette="muted")
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass


# ── 30dk feature tablosu ─────────────────────────────────────────────

def plot_feature_distributions(feat: pd.DataFrame):
    """Temel feature histogramları."""
    cols = ["kayip_kazanc", "kk_oran", "doluluk_oran", "tx_ue1t_fark", "su_diff", "gecikme_dk"]
    cols = [c for c in cols if c in feat.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 7))
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, cols):
        data = feat[col].dropna()
        if col == "kk_oran":
            data = data.clip(-0.3, 0.3)
        ax.hist(data, bins=50, edgecolor="k", alpha=0.75)
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Feature dağılımları (30dk)", y=1.01)
    _show()


def plot_null_rates(feat: pd.DataFrame):
    """Null oranı bar chart."""
    null_pct = (feat.isna().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0].head(15)
    if null_pct.empty:
        print("Null yok.")
        return
    fig, ax = plt.subplots(figsize=(10, max(3, len(null_pct) * 0.35)))
    null_pct.plot(kind="barh", ax=ax, color="coral", edgecolor="k")
    ax.set_xlabel("Null (%)")
    ax.set_title("Feature null oranları")
    _show()


def plot_signal_counts(feat: pd.DataFrame):
    """Kural tabanlı sinyal sayıları."""
    sig_cols = [c for c in feat.columns if c.startswith("sinyal_")]
    if not sig_cols:
        print("Sinyal kolonu yok.")
        return
    counts = feat[sig_cols].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="k")
    ax.set_xlabel("Toplam tetiklenme")
    ax.set_title("Kural tabanlı sinyaller")
    _show()


def plot_kk_oran_vs_satis(feat: pd.DataFrame, sample: int = 8000):
    """kk_oran vs pompa_satis scatter."""
    sub = feat.dropna(subset=["kk_oran", "pompa_satis"])
    sub = sub[sub["pompa_satis"] > 1]
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        sub["pompa_satis"], sub["kk_oran"].clip(-0.5, 0.5),
        c=sub["gece"] if "gece" in sub.columns else "steelblue",
        cmap="coolwarm", alpha=0.35, s=8,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("30dk satış (L)")
    ax.set_ylabel("kk_oran")
    ax.set_title("Satış vs kayıp/kazanç oranı")
    if "gece" in sub.columns:
        plt.colorbar(sc, ax=ax, label="gece")
    _show()


def plot_gece_gunduz_box(feat: pd.DataFrame):
    """Gece vs gündüz kayıp/kazanç boxplot."""
    if "gece" not in feat.columns:
        return
    sub = feat[["gece", "kayip_kazanc"]].copy()
    sub["kayip_kazanc"] = sub["kayip_kazanc"].clip(-100, 100)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=sub, x="gece", y="kayip_kazanc", ax=ax,
                palette=["steelblue", "midnightblue"])
    ax.set_xticklabels(["Gündüz", "Gece"])
    ax.set_title("Kayıp/kazanç — gece vs gündüz")
    ax.axhline(0, color="r", lw=0.5)
    _show()


def plot_hourly_kk(feat: pd.DataFrame):
    """Saat bazında ortalama kk_oran."""
    if "saat" not in feat.columns:
        feat = feat.copy()
        feat["saat"] = feat["saat_1"].dt.hour
    hourly = feat.groupby("saat")["kayip_kazanc"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    hourly.plot(kind="bar", ax=ax, color="teal", edgecolor="k")
    ax.axhline(0, color="r", lw=0.5)
    ax.set_xlabel("Saat")
    ax.set_ylabel("Ort. kayıp/kazanç (L)")
    ax.set_title("Saatlik ortalama kayıp/kazanç")
    _show()


def plot_doluluk_vs_kayip(feat: pd.DataFrame, sample: int = 5000):
    """Doluluk oranı vs kayıp scatter."""
    sub = feat.dropna(subset=["doluluk_oran", "kayip_kazanc"])
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sub["doluluk_oran"], sub["kayip_kazanc"].clip(-80, 80),
               alpha=0.25, s=6, c="purple")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Doluluk oranı")
    ax.set_ylabel("Kayıp/kazanç (L)")
    ax.set_title("Tank doluluğu vs kayıp/kazanç")
    _show()


# ── Günlük feature ───────────────────────────────────────────────────

def plot_daily_fark_alarm(feat_day: pd.DataFrame):
    """Alarm vs normal günlük fark dağılımı."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    feat_day["fark"].hist(bins=60, ax=axes[0], edgecolor="k", alpha=0.75)
    axes[0].set_title("Günlük fark dağılımı")
    for alarm, color in [(0, "steelblue"), (1, "crimson")]:
        d = feat_day[feat_day["alarm"] == alarm]["fark"]
        axes[1].hist(d, bins=40, alpha=0.6, label=f"alarm={alarm}", color=color, edgecolor="k")
    axes[1].legend()
    axes[1].set_title("Fark — alarm vs normal")
    _show()


def plot_daily_fark_vs_satis(feat_day: pd.DataFrame):
    """Günlük fark vs satış."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = feat_day["alarm"].map({0: "steelblue", 1: "crimson"})
    ax.scatter(feat_day["satis"], feat_day["fark"], c=colors, alpha=0.5, s=20)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Günlük satış (L)")
    ax.set_ylabel("Fark (L)")
    ax.set_title("Fark vs satış (kırmızı=alarm)")
    _show()


# ── Manifold ─────────────────────────────────────────────────────────

def plot_manifold_correlations(feat_man: pd.DataFrame):
    """Manifold çift korelasyon bar chart."""
    if feat_man.empty:
        return
    feat_man = feat_man.copy()
    feat_man["label"] = (
        feat_man["istasyon_kodu"] + " T" + feat_man["tank_a"].astype(str)
        + "↔T" + feat_man["tank_b"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["crimson" if c < -0.5 else "steelblue" for c in feat_man["kk_corr"]]
    ax.barh(feat_man["label"], feat_man["kk_corr"], color=colors, edgecolor="k")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Kayıp/kazanç korelasyonu")
    ax.set_title("Manifold çift korelasyonları (ters yön = sızıntı imzası)")
    _show()


# ── Gün 8 / features.csv (etiketli) ──────────────────────────────────

def plot_category_feature_means(features: pd.DataFrame, check_cols: list):
    """Kategori bazında feature ortalamaları heatmap."""
    sub = features[features["anomali_kategorisi"].notna()]
    if sub.empty:
        return
    cols = [c for c in check_cols if c in features.columns]
    ozet = sub.groupby("anomali_kategorisi")[cols].mean()
    # en sık kategoriler
    top_cat = sub["anomali_kategorisi"].value_counts().head(10).index
    ozet = ozet.loc[[c for c in top_cat if c in ozet.index]]
    fig, ax = plt.subplots(figsize=(12, max(4, len(ozet) * 0.45)))
    sns.heatmap(ozet.round(3), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Kategori × feature ortalama (sağlık kontrolü)")
    _show()


def plot_anomaly_rate_by_tank(features: pd.DataFrame):
    """Tank bazında anomali oranı."""
    if "anomali_etiketi" not in features.columns:
        return
    rate = features.groupby(["istasyon_kodu", "tank_no"])["anomali_etiketi"].mean()
    rate = rate.sort_values(ascending=False).head(12)
    labels = [f"{i}/T{t}" for i, t in rate.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels, rate.values, color="crimson", edgecolor="k", alpha=0.8)
    ax.set_xlabel("Anomali oranı")
    ax.set_title("En yüksek anomali oranı — tank (top 12)")
    ax.invert_yaxis()
    _show()


def plot_rolling_ma_example(ue1t: pd.DataFrame, ist: str = "IST_001", tank: int = 1):
    """Tek tank rolling MA örneği (Gün 8)."""
    g = ue1t[(ue1t["istasyon_kodu"] == ist) & (ue1t["tank_no"] == tank)].sort_values("saat_1")
    if g.empty or "kayip_kazanc_ma_48" not in g.columns:
        return
    g = g.iloc[:48 * 14]  # 2 hafta
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(g["saat_1"], g["kayip_kazanc"], alpha=0.3, lw=0.8, label="ham")
    ax.plot(g["saat_1"], g["kayip_kazanc_ma_3"], lw=1, label="MA-3")
    ax.plot(g["saat_1"], g["kayip_kazanc_ma_48"], lw=1.5, label="MA-48")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    ax.set_title(f"{ist} T{tank} — kayıp/kazanç ve hareketli ortalamalar")
    _show()


def plot_kum_egim_distribution(ue1t: pd.DataFrame):
    """Kümülatif eğim dağılımı."""
    if "kum_egim" not in ue1t.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ue1t["kum_egim"].dropna().hist(bins=60, ax=axes[0], edgecolor="k", alpha=0.75)
    axes[0].set_title("kum_egim dağılımı")
    ue1t["kum_egim_r2"].dropna().hist(bins=60, ax=axes[1], edgecolor="k", alpha=0.75, color="coral")
    axes[1].set_title("kum_egim_r2 dağılımı")
    _show()


def plot_vardiya_counts(ue1t: pd.DataFrame):
    """Vardiya dağılımı pie + ortalama kayıp."""
    if "vardiya" not in ue1t.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ue1t["vardiya"].value_counts().plot(kind="pie", ax=axes[0], autopct="%1.0f%%")
    axes[0].set_title("Vardiya dağılımı")
    ue1t.groupby("vardiya")["kayip_kazanc"].mean().plot(
        kind="bar", ax=axes[1], color="steelblue", edgecolor="k"
    )
    axes[1].axhline(0, color="r", lw=0.5)
    axes[1].set_title("Ort. kayıp/kazanç — vardiya")
    _show()


def plot_sicaklik_features(ue1t: pd.DataFrame, sample: int = 5000):
    """Sıcaklık feature scatter'ları."""
    cols = ["sicaklik_fark", "sicaklik_gunluk_sapma", "sicaklik_kayip_korelasyon"]
    cols = [c for c in cols if c in ue1t.columns]
    if not cols:
        return
    sub = ue1t.dropna(subset=cols[:1]).sample(min(sample, len(ue1t)), random_state=42)
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.scatter(sub[col], sub["kayip_kazanc"].clip(-50, 50), alpha=0.2, s=5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(col)
        ax.set_ylabel("kayip_kazanc")
    fig.suptitle("Sıcaklık feature vs kayıp/kazanç", y=1.02)
    _show()


def plot_manifold_ters_yon(ue1t: pd.DataFrame):
    """Manifold ters yön bayrağı dağılımı."""
    if "es_tank_ters_yon" not in ue1t.columns:
        return
    man = ue1t[ue1t["is_manifold"] == 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    man.groupby("istasyon_kodu")["es_tank_ters_yon"].mean().plot(
        kind="bar", ax=ax, color="purple", edgecolor="k"
    )
    ax.set_ylabel("Ters yön oranı")
    ax.set_title("Manifold tanklarda eş-tank ters yön oranı")
    _show()


# ── Master pipeline ek grafikler ─────────────────────────────────────

def plot_data_inventory(dfs: dict):
    """Kaynak tablo boyutları."""
    sizes = {k: v.shape[0] for k, v in dfs.items()}
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.Series(sizes).sort_values().plot(kind="barh", ax=ax, color="teal", edgecolor="k")
    ax.set_xlabel("Satır sayısı")
    ax.set_title("Veri katmanları — kayıt sayısı")
    _show()


def plot_manifold_n_donem(feat_man: pd.DataFrame):
    """Manifold çiftlerde ortak dönem sayısı."""
    if feat_man.empty:
        return
    feat_man = feat_man.copy()
    feat_man["label"] = (
        feat_man["istasyon_kodu"] + " T" + feat_man["tank_a"].astype(str)
        + "↔T" + feat_man["tank_b"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(feat_man["label"], feat_man["n_donem"], color="slateblue", edgecolor="k")
    ax.set_xlabel("Ortak 30dk dönem")
    ax.set_title("Manifold çiftleri — veri kapsamı")
    _show()


def plot_daily_rollup(feat_day: pd.DataFrame):
    """Günlük fark vs UE1T rollup karşılaştırması."""
    cols = ["fark", "ue1t_kk_toplam", "ue1t_gece_kk"]
    cols = [c for c in cols if c in feat_day.columns]
    if len(cols) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    if "ue1t_kk_toplam" in feat_day.columns:
        axes[0].scatter(feat_day["ue1t_kk_toplam"], feat_day["fark"], alpha=0.4, s=15, c="steelblue")
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].axvline(0, color="k", lw=0.5)
        axes[0].set_xlabel("UE1T kk toplam (L)")
        axes[0].set_ylabel("Günlük fark (L)")
        axes[0].set_title("Günlük fark vs UE1T rollup")
    if "sel_asildi" in feat_day.columns:
        sns.boxplot(
            data=feat_day, x="sel_asildi", y="fark", ax=axes[1],
            palette=["steelblue", "crimson"],
        )
        axes[1].set_xticklabels(["SEL altı", "SEL üstü"])
        axes[1].set_title("Fark — SEL eşiği")
    _show()


def plot_merged_overlay(feat_merged: pd.DataFrame, sample: int = 6000):
    """Birleşik tabloda günlük vs 30dk sinyaller."""
    sub = feat_merged.dropna(subset=["kayip_kazanc", "fark_abs"], how="any")
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sc = axes[0].scatter(
        sub["kayip_kazanc"].clip(-80, 80), sub["fark_abs"],
        c=sub["alarm"] if "alarm" in sub.columns else "steelblue",
        cmap="coolwarm", alpha=0.35, s=8,
    )
    axes[0].set_xlabel("30dk kayıp/kazanç (L)")
    axes[0].set_ylabel("|Günlük fark| (L)")
    axes[0].set_title("Birleşik tablo — 30dk kk vs günlük fark")
    if "alarm" in sub.columns:
        plt.colorbar(sc, ax=axes[0], label="alarm")
    if "ue1t_satisiz_oran" in sub.columns:
        axes[1].scatter(
            sub["ue1t_satisiz_oran"], sub["kayip_kazanc"].clip(-80, 80),
            alpha=0.25, s=6, c="darkorange",
        )
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xlabel("Günlük satışsız oran")
        axes[1].set_ylabel("30dk kayıp/kazanç")
        axes[1].set_title("Satışsız oran vs anlık kk")
    _show()


def plot_feature_corr_heatmap(feat: pd.DataFrame, cols: list | None = None):
    """Seçili feature korelasyon matrisi."""
    if cols is None:
        cols = [
            "kayip_kazanc", "kk_oran", "pompa_satis", "doluluk_oran",
            "tx_ue1t_fark", "su_diff", "fark_abs", "ue1t_kk_toplam",
        ]
    cols = [c for c in cols if c in feat.columns]
    if len(cols) < 3:
        return
    corr = feat[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr.round(2), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature korelasyon matrisi")
    _show()


def plot_export_summary(paths: dict):
    """Export edilen CSV boyutları."""
    rows = []
    for name, p in paths.items():
        p = Path(p) if not isinstance(p, Path) else p
        if p.exists():
            rows.append({"dosya": p.name, "mb": p.stat().st_size / 1e6})
    if not rows:
        print("Export dosyası yok.")
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.barh(df["dosya"], df["mb"], color="forestgreen", edgecolor="k")
    ax.set_xlabel("Boyut (MB)")
    ax.set_title("Export edilen dataset dosyaları")
    _show()
