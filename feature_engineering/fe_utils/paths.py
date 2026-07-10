# -*- coding: utf-8 -*-
"""Feature engineering yolları — EDA utils ile uyumlu."""

import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
PROJECT = FE_ROOT.parent
EDA_ROOT = PROJECT / "eda"


def ensure_paths():
    for p in (str(EDA_ROOT), str(FE_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def fe_figures(section: str = "overview") -> Path:
    out = FE_ROOT / "output" / "figures" / section
    out.mkdir(parents=True, exist_ok=True)
    return out


def fe_datasets() -> Path:
    out = FE_ROOT / "output" / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    return out
