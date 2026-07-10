# -*- coding: utf-8 -*-
"""Katmanlar arası tutarlılık ve mutabakat kontrolleri."""

import pandas as pd
import numpy as np


def check_ue1t_balance(ue1t: pd.DataFrame) -> pd.DataFrame:
    """
    UE1T iç mutabakat:
    donem_sonu = donem_basi + dolum - satis + kayip
    """
    residual = (
        ue1t["donem_sonu_stok"]
        - (
            ue1t["donem_basi_stok"]
            + ue1t["tanka_dolum"]
            - ue1t["pompa_satis"]
            + ue1t["kayip_kazanc"]
        )
    )
    out = ue1t.copy()
    out["mutabakat_residual"] = residual
    return out


def check_continuity(ue1t: pd.DataFrame) -> pd.DataFrame:
    """Bir dönemin sonu = sonraki dönemin başı (kesinti hariç)."""
    u = ue1t.sort_values(["istasyon_kodu", "tank_no", "saat_1"]).copy()
    u["onceki_son"] = u.groupby(["istasyon_kodu", "tank_no"])["donem_sonu_stok"].shift(1)
    u["sureklilik_fark"] = u["donem_basi_stok"] - u["onceki_son"]
    return u.dropna(subset=["onceki_son"])


def check_tx_to_ue1t(
    transactions: pd.DataFrame,
    ue1t: pd.DataFrame,
) -> pd.DataFrame:
    """Tekil satışlar 30 dk'ya toplanınca pompa_satis'e eşit mi?"""
    tx = transactions[transactions["tank_no"].notna()].copy()
    tx["pencere"] = tx["satis_zamani"].dt.floor("30min")
    agg = (
        tx.groupby(["istasyon_kodu", "tank_no", "pencere"])["litre"]
        .sum()
        .reset_index()
    )
    m = ue1t.merge(
        agg,
        left_on=["istasyon_kodu", "tank_no", "saat_1"],
        right_on=["istasyon_kodu", "tank_no", "pencere"],
        how="left",
    )
    m["tx_toplam"] = m["litre"].fillna(0)
    m["tx_ue1t_fark"] = m["pompa_satis"] - m["tx_toplam"]
    return m


def check_ue1t_to_daily(ue1t: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """30 dk toplamları günlük özete eşit mi?"""
    u = ue1t.copy()
    u["tarih"] = u["saat_1"].dt.normalize()
    du = u.groupby(["istasyon_kodu", "tank_no", "tarih"], as_index=False).agg(
        ue1t_satis=("pompa_satis", "sum"),
        ue1t_dolum=("tanka_dolum", "sum"),
        ue1t_acilis=("donem_basi_stok", "first"),
        ue1t_kapanis=("donem_sonu_stok", "last"),
    )
    m = daily.merge(du, on=["istasyon_kodu", "tank_no", "tarih"], how="left")
    m["satis_fark"] = m["satis"] - m["ue1t_satis"]
    m["dolum_fark"] = m["dolum"] - m["ue1t_dolum"]
    m["acilis_fark"] = m["acilis"] - m["ue1t_acilis"]
    m["kapanis_fark"] = m["kapanis"] - m["ue1t_kapanis"]
    return m


def check_deliveries_to_daily(
    deliveries: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Dolum kayıtları günlük doluma eşit mi?"""
    d = deliveries.copy()
    d["tarih"] = d["dolum_baslangic"].dt.normalize()
    dd = d.groupby(["istasyon_kodu", "tank_no", "tarih"], as_index=False)[
        "dolum_net"
    ].sum()
    m = daily.merge(dd, on=["istasyon_kodu", "tank_no", "tarih"], how="left")
    m["dolum_net"] = m["dolum_net"].fillna(0)
    m["delivery_fark"] = m["dolum"] - m["dolum_net"]
    return m


def run_all_checks(dfs: dict) -> dict:
    """Tüm kontrolleri çalıştırır, özet metrik döner."""
    ue1t = check_ue1t_balance(dfs["ue1t_30min"])
    cont = check_continuity(dfs["ue1t_30min"])
    txu = check_tx_to_ue1t(dfs["transactions"], dfs["ue1t_30min"])
    ud = check_ue1t_to_daily(dfs["ue1t_30min"], dfs["daily"])
    dd = check_deliveries_to_daily(dfs["deliveries"], dfs["daily"])

    return {
        "ue1t_balance": ue1t,
        "continuity": cont,
        "tx_to_ue1t": txu,
        "ue1t_to_daily": ud,
        "deliveries_to_daily": dd,
        "summary": {
            "mutabakat_max_abs": float(ue1t["mutabakat_residual"].abs().max()),
            "sureklilik_gt_001": int((cont["sureklilik_fark"].abs() > 0.01).sum()),
            "tx_ue1t_gt_001": int((txu["tx_ue1t_fark"].abs() > 0.01).sum()),
            "daily_satis_max_fark": float(ud["satis_fark"].abs().max()),
            "delivery_fark_gt_001": int((dd["delivery_fark"].abs() > 0.01).sum()),
        },
    }
