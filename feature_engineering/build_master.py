#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature engineering master notebook oluşturur."""

import json
from pathlib import Path

FE = Path(__file__).resolve().parent
OUT = FE / "notebooks" / "FE_Master.ipynb"

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})

def code(s):
    cells.append({
        "cell_type": "code", "metadata": {}, "source": s,
        "outputs": [], "execution_count": None,
    })

md("""# Feature Engineering — Birleşik Master

EDA bulgularından **ML-ready** feature tabloları üretir.

| Adım | Çıktı |
|------|-------|
| 1 | 30dk feature tablosu (`features_30min.csv`) |
| 2 | Günlük feature tablosu (`features_daily.csv`) |
| 3 | Manifold çift metrikleri |
| 4 | Birleşik 30dk tablo |
| 5 | Görselleştirmeler → `output/figures/` |

> `data/ground_truth/` sadece model değerlendirmesinde kullanılır — feature üretiminde açılmaz.
""")

code("""# Kurulum
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'feature_engineering' and (ROOT / 'feature_engineering').exists():
    ROOT = ROOT / 'feature_engineering'
EDA = ROOT.parent / 'eda'
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all, DATA_DIR
from fe_utils.features import (
    build_features_30min, build_features_daily,
    build_manifold_features, merge_feature_layers, export_features,
)
from fe_utils.fe_plots import plot_feature_overview, plot_null_heatmap, plot_signal_counts
from fe_utils.paths import fe_datasets, fe_figures

print('Veri:', DATA_DIR)
print('Dataset çıktısı:', fe_datasets())
print('Grafik çıktısı:', fe_figures())""")

md("## 1 — Veri yükleme")
code("""dfs = load_all()
print({k: v.shape for k, v in dfs.items()})""")

md("## 2 — 30 dakika feature tablosu")
code("""feat_30 = build_features_30min(dfs)
print('Shape:', feat_30.shape)
print('Kolon sayısı:', feat_30.shape[1])
feat_30.head()""")

code("""# Feature özeti
feat_cols = ['kayip_kazanc','kk_oran','doluluk_oran','tx_ue1t_fark','su_diff','gecikme_dk']
feat_cols = [c for c in feat_cols if c in feat_30.columns]
feat_30[feat_cols].describe().round(3)""")

md("## 3 — Günlük feature tablosu")
code("""feat_day = build_features_daily(dfs)
print('Shape:', feat_day.shape)
feat_day[['fark','fark_abs','sel_asildi','ue1t_kk_toplam','ue1t_gece_kk']].describe().round(2)""")

md("## 4 — Manifold çift metrikleri")
code("""feat_man = build_manifold_features(feat_30, dfs)
display(feat_man)""")

md("## 5 — Katman birleştirme (30dk + günlük)")
code("""feat_merged = merge_feature_layers(feat_30, feat_day)
print('Birleşik shape:', feat_merged.shape)
print('Ek kolonlar:', set(feat_merged.columns) - set(feat_30.columns))""")

md("## 6 — Export (CSV)")
code("""paths = export_features(feat_30, feat_day, feat_man, feat_merged)
for name, p in paths.items():
    print(f'{name}: {p}')""")

md("## 7 — Görselleştirmeler")
code("""p1 = plot_feature_overview(feat_30)
p2 = plot_null_heatmap(feat_30)
p3 = plot_signal_counts(feat_30)
print('Kaydedildi:', p1, p2, p3, sep='\\n')""")

md("""## Özet

**Üretilen dosyalar** (`feature_engineering/output/datasets/`):
- `features_30min.csv` — ana ML feature tablosu (138K satır)
- `features_daily.csv` — günlük alarm/fark feature'ları
- `features_manifold.csv` — manifold çift korelasyonları
- `features_merged_30min.csv` — 30dk + günlük birleşik

**Sonraki adım:** Zaman bazlı train/test split + `ground_truth` ile model eğitimi.
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Oluşturuldu:", OUT)
