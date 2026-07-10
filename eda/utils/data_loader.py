# -*- coding: utf-8 -*-
"""Wetstock sentetik veri ambarı — yükleme ve filtreleme yardımcıları."""

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd

# eda/ klasöründen ../data/ yoluna gider
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_all(data_dir: Optional[Union[Path, str]] = None) -> Dict[str, pd.DataFrame]:
    """8 operasyonel tabloyu sözlük olarak yükler."""
    root = Path(data_dir) if data_dir else DATA_DIR
    return {
        "stations": pd.read_csv(root / "stations.csv"),
        "tanks": pd.read_csv(root / "tanks.csv"),
        "mapping": pd.read_csv(root / "mapping.csv"),
        "transactions": pd.read_csv(
            root / "transactions.csv", parse_dates=["satis_zamani"]
        ),
        "deliveries": pd.read_csv(
            root / "deliveries.csv",
            parse_dates=["dolum_baslangic", "dolum_bitis", "merkeze_gelis_tarihi"],
        ),
        "inventory_30min": pd.read_csv(
            root / "inventory_30min.csv",
            parse_dates=["envanter_tarihi", "merkeze_gelis_tarihi"],
        ),
        "ue1t_30min": pd.read_csv(
            root / "ue1t_30min.csv", parse_dates=["saat_1", "saat_2"]
        ),
        "daily": pd.read_csv(root / "daily.csv", parse_dates=["tarih"]),
    }


def filter_tank(
    df: pd.DataFrame,
    istasyon_kodu: str,
    tank_no: int,
) -> pd.DataFrame:
    """istasyon_kodu + tank_no ile filtreler."""
    return df[
        (df["istasyon_kodu"] == istasyon_kodu) & (df["tank_no"] == tank_no)
    ].copy()


def summary_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Her tablo için satır/kolon/null özeti."""
    rows = []
    for name, df in dfs.items():
        null_cols = df.isna().sum()
        null_cols = null_cols[null_cols > 0]
        rows.append({
            "tablo": name,
            "satir": len(df),
            "kolon": df.shape[1],
            "null_kolon_sayisi": len(null_cols),
            "toplam_null": int(df.isna().sum().sum()),
            "null_detay": null_cols.to_dict() if len(null_cols) else {},
        })
    return pd.DataFrame(rows)


def merge_ue1t_inventory(
    ue1t: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """UE1T ile envanteri saat_2 = envanter_tarihi üzerinden birleştirir."""
    return ue1t.merge(
        inventory,
        left_on=["istasyon_kodu", "tank_no", "saat_2"],
        right_on=["istasyon_kodu", "tank_no", "envanter_tarihi"],
        how="left",
        suffixes=("_ue1t", "_inv"),
    )
