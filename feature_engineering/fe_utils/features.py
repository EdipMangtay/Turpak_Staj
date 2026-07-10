# -*- coding: utf-8 -*-
"""Wetstock feature tabloları — EDA bulgularından birleşik dataset."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .paths import ensure_paths

ensure_paths()
from utils.data_loader import load_all  # noqa: E402
from utils.validation import check_tx_to_ue1t  # noqa: E402


def build_features_30min(dfs: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """UE1T + envanter + TX aggregate → 30dk feature tablosu."""
    if dfs is None:
        dfs = load_all()

    ue = dfs["ue1t_30min"].copy()
    inv = dfs["inventory_30min"].copy()
    tx = dfs["transactions"]

    # UE1T türevleri
    ue["saat"] = ue["saat_1"].dt.hour
    ue["gece"] = ue["saat"].between(0, 5).astype(int)
    ue["satisiz"] = (ue["pompa_satis"] == 0).astype(int)
    ue["kk_oran"] = np.where(
        ue["pompa_satis"] > 1,
        ue["kayip_kazanc"] / ue["pompa_satis"],
        np.nan,
    )
    ue = ue.sort_values(["istasyon_kodu", "tank_no", "saat_1"])
    ue["stok_diff"] = ue.groupby(["istasyon_kodu", "tank_no"])["donem_sonu_stok"].diff()
    ue["dsicaklik"] = ue.groupby(["istasyon_kodu", "tank_no"])["sicaklik"].diff()

    # Envanter join
    inv = inv.rename(columns={
        "sicaklik": "sicaklik_inv",
        "envanter_tarihi": "saat_2",
    })
    inv_cols = [
        "istasyon_kodu", "tank_no", "saat_2",
        "urun_miktari_brut", "urun_miktari_net", "su_seviyesi_cm", "su_miktari",
        "sicaklik_inv", "merkeze_gelis_tarihi",
    ]
    inv_sub = inv[[c for c in inv_cols if c in inv.columns]]
    feat = ue.merge(inv_sub, on=["istasyon_kodu", "tank_no", "saat_2"], how="left")
    feat["brut_net_fark"] = feat["urun_miktari_brut"] - feat["urun_miktari_net"]
    feat["gecikme_dk"] = (
        (feat["merkeze_gelis_tarihi"] - feat["saat_2"]).dt.total_seconds() / 60
    )

    # Su sıçraması
    feat = feat.sort_values(["istasyon_kodu", "tank_no", "saat_1"])
    feat["su_diff"] = feat.groupby(["istasyon_kodu", "tank_no"])["su_seviyesi_cm"].diff()

    # TX → UE1T fark
    txu = check_tx_to_ue1t(tx, dfs["ue1t_30min"])
    feat = feat.merge(
        txu[["istasyon_kodu", "tank_no", "saat_1", "tx_ue1t_fark", "tx_toplam"]],
        on=["istasyon_kodu", "tank_no", "saat_1"],
        how="left",
    )
    feat["tx_ue1t_fark"] = feat["tx_ue1t_fark"].fillna(0)

    # Tank meta
    tanks = dfs["tanks"][[
        "istasyon_kodu", "tank_no", "kapasite", "akaryakit_turu",
        "is_manifold", "manifold_grup_no", "bolmeli", "bolme_grup_no",
    ]]
    feat = feat.merge(tanks, on=["istasyon_kodu", "tank_no"], how="left")
    feat["doluluk_oran"] = feat["donem_sonu_stok"] / feat["kapasite"]

    # Kural tabanlı sinyaller
    feat["sinyal_satisiz_kayip"] = (
        (feat["pompa_satis"] == 0) & (feat["kayip_kazanc"] < -5)
    ).astype(int)
    feat["sinyal_gece_dusus"] = (
        (feat["gece"] == 1) & (feat["pompa_satis"] == 0) & (feat["kayip_kazanc"] < -20)
    ).astype(int)
    feat["sinyal_stok_dondu"] = (
        (feat["pompa_satis"] > 30) & (feat["stok_diff"].abs() < 0.01)
    ).astype(int)
    feat["sinyal_su_spike"] = (feat["su_diff"] > 0.05).astype(int)

    return feat


def build_features_daily(dfs: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """Daily tablosu + UE1T rollup → günlük feature tablosu."""
    if dfs is None:
        dfs = load_all()

    daily = dfs["daily"].copy()
    ue = dfs["ue1t_30min"].copy()
    ue["tarih"] = ue["saat_1"].dt.normalize()
    ue["satisiz"] = (ue["pompa_satis"] == 0).astype(int)
    ue["gece"] = ue["saat_1"].dt.hour.between(0, 5)

    roll = ue.groupby(["istasyon_kodu", "tank_no", "tarih"], as_index=False).agg(
        ue1t_kk_toplam=("kayip_kazanc", "sum"),
        ue1t_kk_std=("kayip_kazanc", "std"),
        ue1t_satisiz_oran=("satisiz", "mean"),
    )
    gece = ue[ue["gece"]].groupby(
        ["istasyon_kodu", "tank_no", "tarih"], as_index=False
    ).agg(ue1t_gece_kk=("kayip_kazanc", "sum"))
    roll = roll.merge(gece, on=["istasyon_kodu", "tank_no", "tarih"], how="left")

    feat = daily.merge(roll, on=["istasyon_kodu", "tank_no", "tarih"], how="left")
    feat["fark_abs"] = feat["fark"].abs()
    feat["sel_asildi"] = (feat["fark_abs"] > feat["sel"]).astype(int)
    feat["fark_satis_oran"] = np.where(feat["satis"] > 0, feat["fark"] / feat["satis"], np.nan)
    return feat


def build_manifold_features(
    feat_30min: pd.DataFrame,
    dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Manifold çiftler için ters yönlü kayıp/kazanç korelasyonu."""
    if dfs is None:
        dfs = load_all()
    tanks = dfs["tanks"]
    man = tanks[tanks["is_manifold"] == 1]

    rows = []
    for (ist, grp), g in man.groupby(["istasyon_kodu", "manifold_grup_no"]):
        tn = sorted(g["tank_no"].tolist())
        if len(tn) != 2:
            continue
        a, b = tn
        ua = feat_30min[(feat_30min.istasyon_kodu == ist) & (feat_30min.tank_no == a)].set_index("saat_1")["kayip_kazanc"]
        ub = feat_30min[(feat_30min.istasyon_kodu == ist) & (feat_30min.tank_no == b)].set_index("saat_1")["kayip_kazanc"]
        m = pd.concat([ua, ub], axis=1, keys=["a", "b"]).dropna()
        corr = m["a"].corr(m["b"]) if len(m) > 50 else np.nan
        rows.append({
            "istasyon_kodu": ist,
            "manifold_grup_no": grp,
            "tank_a": a,
            "tank_b": b,
            "kk_corr": corr,
            "n_donem": len(m),
        })
    return pd.DataFrame(rows)


def merge_feature_layers(
    feat_30min: pd.DataFrame,
    feat_daily: pd.DataFrame,
) -> pd.DataFrame:
    """30dk tabloyu günlük feature'larla zenginleştir (aynı gün)."""
    f = feat_30min.copy()
    f["tarih"] = f["saat_1"].dt.normalize()
    daily_cols = [c for c in feat_daily.columns if c not in (
        "istasyon_kodu", "tank_no", "tarih", "acilis", "kapanis",
        "satis", "dolum", "fark", "oran", "sel", "alarm", "azalma_miktari",
        "akaryakit_turu",
    )]
    dsub = feat_daily[["istasyon_kodu", "tank_no", "tarih"] + daily_cols]
    return f.merge(dsub, on=["istasyon_kodu", "tank_no", "tarih"], how="left", suffixes=("", "_gun"))


def export_features(
    feat_30min: pd.DataFrame,
    feat_daily: pd.DataFrame,
    feat_manifold: pd.DataFrame,
    merged: pd.DataFrame,
    out_dir=None,
) -> dict:
    """CSV olarak kaydet."""
    from .paths import fe_datasets

    out = out_dir or fe_datasets()
    paths = {
        "features_30min": out / "features_30min.csv",
        "features_daily": out / "features_daily.csv",
        "features_manifold": out / "features_manifold.csv",
        "features_merged_30min": out / "features_merged_30min.csv",
    }
    feat_30min.to_csv(paths["features_30min"], index=False)
    feat_daily.to_csv(paths["features_daily"], index=False)
    feat_manifold.to_csv(paths["features_manifold"], index=False)
    # merged büyük — sadece önemli kolonlar + sample için tam kayıt
    merged.to_csv(paths["features_merged_30min"], index=False)
    return paths
