#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tek master build — EDA_Master.ipynb ve FE_Master.ipynb oluşturur.

Kullanım:
    python build_master.py          # ikisini birden
    python build_master.py --eda    # sadece EDA
    python build_master.py --fe     # sadece FE
"""

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
EDA = PROJECT / "eda"
FE = PROJECT / "feature_engineering"

NB_META = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
}


def _md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


def _write_nb(path: Path, cells: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {**NB_META, "cells": cells}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Oluşturuldu: {path} ({len(cells)} hücre)")


def build_eda():
    """01–10 EDA notebook'larını EDA_Master.ipynb'de birleştirir."""
    notebooks_dir = EDA / "notebooks"
    out = EDA / "EDA_Master.ipynb"

    order = [
        ("01_veri_semasi_ve_ozet.ipynb", "01_schema", "BÖLÜM 1 — Veri şeması ve özet"),
        ("02_katman_tutarliligi.ipynb", "02_validation", "BÖLÜM 2 — Katman tutarlılığı"),
        ("03_null_ve_eksik_veri.ipynb", "03_nulls", "BÖLÜM 3 — Null ve eksik veri"),
        ("04_tek_tank_derinlemesine.ipynb", "04_single_tank", "BÖLÜM 4 — Tek tank derinlemesine"),
        ("05_gunluk_alarm_ve_fark.ipynb", "05_daily_alarm", "BÖLÜM 5 — Günlük alarm ve fark"),
        ("06_envanter_su_sicaklik.ipynb", "06_inventory", "BÖLÜM 6 — Envanter, su, sıcaklık"),
        ("07_satis_ve_dolum.ipynb", "07_sales_delivery", "BÖLÜM 7 — Satış ve dolum"),
        ("08_manifold_bolmeli.ipynb", "08_manifold", "BÖLÜM 8 — Manifold ve bölmeli"),
        ("09_anomali_kesfi.ipynb", "09_anomaly", "BÖLÜM 9 — Anomali keşfi"),
        ("10_gun07_derin_eda.ipynb", "10_deep_eda", "BÖLÜM 10 — Derin EDA (Gün 7)"),
    ]

    setup = """# ============================================================
# KURULUM — tüm bölümler bu ortamı paylaşır
# ============================================================
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda' and (ROOT / 'eda').exists():
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import (
    load_all, filter_tank, merge_ue1t_inventory, summary_table, DATA_DIR,
)
from utils.validation import run_all_checks, check_tx_to_ue1t
from utils.plots import setup_style, save_fig, set_section
from utils.paths import figures_dir

setup_style()
sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', 40)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

print('Veri klasörü:', DATA_DIR)
print('Grafik çıktısı:', figures_dir())
dfs = load_all()
ue = dfs['ue1t_30min']
tanks = dfs['tanks']
inv = dfs['inventory_30min']
daily = dfs['daily']
tx = dfs['transactions']
deliv = dfs['deliveries']
mapping = dfs['mapping']
print('Yüklendi:', {k: v.shape for k, v in dfs.items()})"""

    def is_setup_only(src: str) -> bool:
        lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
        if not lines:
            return True
        skip = (
            "import sys", "import pandas", "import numpy", "import matplotlib",
            "import seaborn", "from pathlib", "ROOT =", "if ROOT.name", "elif ROOT",
            "sys.path.insert", "from utils.", "setup_style()", "sns.set_theme",
            "pd.set_option", "print('Veri", "dfs = load_all()", "list(dfs.keys())",
            "daily = load_all()", "daily = dfs", "ue = dfs", "tanks = dfs",
            "inv = dfs", "tx = dfs", "deliv = dfs",
        )
        return all(any(l.lstrip().startswith(p) for p in skip) for l in lines)

    def inject_fig_save(src: str, section: str, fig_idx: list) -> str:
        if "plt.show()" not in src:
            return src
        fig_idx[0] += 1
        rep = f"set_section('{section}')\nsave_fig(name='{section}_{fig_idx[0]:02d}')"
        return src.replace("plt.show()", rep)

    cells = [
        _md(
            """# Wetstock EDA — Birleşik Master Notebook

Tüm keşifsel analiz adımları **tek dosyada**, bölüm bölüm.

| Bölüm | Konu |
|-------|------|
| 1–10 | Şema → tutarlılık → null → tank → alarm → envanter → satış → manifold → anomali → derin EDA |

> Grafikler `eda/output/figures/<bölüm>/` altına kaydedilir.
> `data/ground_truth/` bu aşamada **kullanılmaz**.

---"""
        ),
        _code(setup),
    ]

    for fname, section, title in order:
        path = notebooks_dir / fname
        if not path.exists():
            print("Atlanıyor (yok):", fname)
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        cells.append(_md(f"---\n## {title}\n\nKaynak: `notebooks/{fname}`\n"))
        fig_idx = [0]
        for cell in nb["cells"]:
            if cell["cell_type"] == "markdown":
                src = "".join(cell["source"])
                if src.startswith("# ") and "—" in src[:40]:
                    continue
                if src.strip():
                    cells.append(_md(src))
            elif cell["cell_type"] == "code":
                src_orig = "".join(cell["source"])
                if is_setup_only(src_orig):
                    continue
                src = src_orig if src_orig.endswith("\n") else src_orig + "\n"
                src = inject_fig_save(src, section, fig_idx)
                cells.append(_code(src))

    _write_nb(out, cells)


def build_fe():
    """FE_Master.ipynb — her adımın altında inline görselleştirme."""
    out = FE / "notebooks" / "FE_Master.ipynb"
    cells = []

    cells.append(_md("""# Feature Engineering — Birleşik Master

EDA bulgularından **ML-ready** feature tabloları. Grafikler her adımın **hemen altında** inline gösterilir.

> Detaylı Gün 8 notebook: `GUN08_feature_engineering.ipynb`
"""))

    cells.append(_code("""# Kurulum
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from fe_utils.fe_plots import (
    setup_fe_style,
    plot_data_inventory,
    plot_feature_distributions, plot_null_rates, plot_signal_counts,
    plot_kk_oran_vs_satis, plot_gece_gunduz_box, plot_hourly_kk,
    plot_doluluk_vs_kayip, plot_daily_fark_alarm, plot_daily_fark_vs_satis,
    plot_daily_rollup, plot_manifold_correlations, plot_manifold_n_donem,
    plot_merged_overlay, plot_feature_corr_heatmap, plot_export_summary,
)

setup_fe_style()
%matplotlib inline
print('Veri:', DATA_DIR)"""))

    steps = [
        ("## 1 — Veri yükleme",
         "dfs = load_all()\nprint({k: v.shape for k, v in dfs.items()})",
         "plot_data_inventory(dfs)"),
        ("## 2 — 30 dk feature tablosu",
         "feat_30 = build_features_30min(dfs)\nprint('Shape:', feat_30.shape)\ndisplay(feat_30.head())",
         """plot_feature_distributions(feat_30)
plot_null_rates(feat_30)
plot_signal_counts(feat_30)"""),
        ("### 2b — 30 dk davranış grafikleri",
         "# Satış, saat ve doluluk ilişkileri",
         """plot_kk_oran_vs_satis(feat_30)
plot_gece_gunduz_box(feat_30)
plot_hourly_kk(feat_30)
plot_doluluk_vs_kayip(feat_30)"""),
        ("## 3 — Günlük feature tablosu",
         "feat_day = build_features_daily(dfs)\nprint('Shape:', feat_day.shape)\ndisplay(feat_day.describe().round(2).T.head(12))",
         """plot_daily_fark_alarm(feat_day)
plot_daily_fark_vs_satis(feat_day)
plot_daily_rollup(feat_day)"""),
        ("## 4 — Manifold metrikleri",
         "feat_man = build_manifold_features(feat_30, dfs)\ndisplay(feat_man)",
         """plot_manifold_correlations(feat_man)
plot_manifold_n_donem(feat_man)"""),
        ("## 5 — Birleştirme",
         "feat_merged = merge_feature_layers(feat_30, feat_day)\nprint('Shape:', feat_merged.shape)",
         """plot_merged_overlay(feat_merged)
plot_feature_corr_heatmap(feat_merged)"""),
        ("## 6 — Export (CSV)",
         "paths = export_features(feat_30, feat_day, feat_man, feat_merged)\nfor n, p in paths.items(): print(n, '→', p.name)",
         "plot_export_summary(paths)"),
    ]

    for title, code_src, viz_src in steps:
        cells.append(_md(title))
        cells.append(_code(code_src))
        if viz_src:
            cells.append(_code(viz_src))

    _write_nb(out, cells)


def build_gun08():
    """GUN08 notebook'unu inline görselleştirmelerle oluşturur."""
    import importlib.util
    spec_path = PROJECT / "scripts" / "archive" / "build_gun8_notebook.py"
    spec = importlib.util.spec_from_file_location("build_gun8", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    parser = argparse.ArgumentParser(description="Master notebook builder")
    parser.add_argument("--eda", action="store_true", help="Sadece EDA_Master")
    parser.add_argument("--fe", action="store_true", help="Sadece FE_Master")
    args = parser.parse_args()

    run_eda = args.eda or not (args.eda or args.fe)
    run_fe = args.fe or not (args.eda or args.fe)

    if run_eda:
        print("\n=== EDA Master ===")
        build_eda()
    if run_fe:
        print("\n=== FE Master ===")
        build_fe()
        print("\n=== GUN08 ===")
        build_gun08()
    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
