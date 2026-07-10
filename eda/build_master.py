#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01–10 EDA notebook'larını tek EDA_Master.ipynb dosyasında birleştirir."""

import json
import re
from pathlib import Path

EDA = Path(__file__).resolve().parent
NOTEBOOKS_DIR = EDA / "notebooks"
OUT = EDA / "EDA_Master.ipynb"

ORDER = [
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

SETUP = """# ============================================================
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

IMPORT_BLOCK = re.compile(
    r"^import sys.*?from utils\.plots import.*?\n(?:setup_style\(\).*?\n)?(?:sns\.set_theme.*?\n)?"
    r"(?:pd\.set_option.*?\n)*"
    r"(?:print\(.*?\n)?(?:dfs = load_all\(\).*?\n)?"
    r"(?:list\(dfs\.keys\(\)\).*?\n)?"
    r"(?:daily = .*?\n)?",
    re.DOTALL | re.MULTILINE,
)


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


def is_setup_only_cell(src: str) -> bool:
    """Tamamen kurulum/import olan hücreleri atla."""
    lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return True
    skip_prefixes = (
        "import sys", "import pandas", "import numpy", "import matplotlib",
        "import seaborn", "from pathlib", "ROOT =", "if ROOT.name", "elif ROOT",
        "sys.path.insert", "from utils.", "setup_style()", "sns.set_theme",
        "pd.set_option", "print('Veri", "dfs = load_all()", "list(dfs.keys())",
        "daily = load_all()", "daily = dfs", "ue = dfs", "tanks = dfs",
        "inv = dfs", "tx = dfs", "deliv = dfs",
    )
    for line in lines:
        stripped = line.lstrip()
        if not any(stripped.startswith(p) for p in skip_prefixes):
            return False
    return True


def inject_fig_save(src: str, section: str, fig_idx: list) -> str:
    """plt.show() sonrasına save_fig ekle."""
    if "plt.show()" not in src:
        return src
    fig_idx[0] += 1
    name = f"{section}_{fig_idx[0]:02d}"
    replacement = f"set_section('{section}')\nsave_fig(name='{name}')"
    return src.replace("plt.show()", replacement)


def load_nb(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    cells = [
        md(
            """# Wetstock EDA — Birleşik Master Notebook

Tüm keşifsel analiz adımları **tek dosyada**, bölüm bölüm.

| Bölüm | Konu |
|-------|------|
| 1 | Veri şeması ve özet |
| 2 | Katman tutarlılığı |
| 3 | Null / eksik veri |
| 4 | Tek tank derinlemesine |
| 5 | Günlük alarm ve fark |
| 6 | Envanter, su, sıcaklık |
| 7 | Satış ve dolum |
| 8 | Manifold ve bölmeli |
| 9 | Anomali keşfi |
| 10 | Derin EDA (Gün 7) |

> Grafikler `eda/output/figures/<bölüm>/` altına kaydedilir.
> `data/ground_truth/` bu aşamada **kullanılmaz**.

---"""
        ),
        code(SETUP),
    ]

    for fname, section, title in ORDER:
        path = NOTEBOOKS_DIR / fname
        if not path.exists():
            print("Atlanıyor (yok):", fname)
            continue
        nb = load_nb(path)
        cells.append(md(f"---\n## {title}\n\nKaynak: `notebooks/{fname}`\n"))
        fig_idx = [0]
        for cell in nb["cells"]:
            if cell["cell_type"] == "markdown":
                src = "".join(cell["source"])
                # İlk notebook başlığını atla (master'da zaten var)
                if src.startswith("# ") and "—" in src[:40]:
                    continue
                if src.strip():
                    cells.append(md(src))
            elif cell["cell_type"] == "code":
                src_orig = "".join(cell["source"])
                if is_setup_only_cell(src_orig):
                    continue
                src = src_orig if src_orig.endswith("\n") else src_orig + "\n"
                src = inject_fig_save(src, section, fig_idx)
                cells.append(code(src))

    nb_out = {
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
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Oluşturuldu: {OUT} ({len(cells)} hücre)")


if __name__ == "__main__":
    main()
