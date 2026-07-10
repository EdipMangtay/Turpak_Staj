# -*- coding: utf-8 -*-
"""Proje yolları — EDA ve feature engineering ortak."""

from pathlib import Path


def project_root() -> Path:
    """Staj/ kök dizini."""
    return Path(__file__).resolve().parents[2]


def eda_root() -> Path:
    return project_root() / "eda"


def fe_root() -> Path:
    return project_root() / "feature_engineering"


def data_dir() -> Path:
    return project_root() / "data"


def figures_dir(module: str = "eda", section: str = "genel") -> Path:
    """
    Görselleştirme çıktı klasörü.
    module: 'eda' | 'fe'
    section: '01_schema', '02_validation', ...
    """
    if module == "fe":
        base = fe_root() / "output" / "figures"
    else:
        base = eda_root() / "output" / "figures"
    out = base / section
    out.mkdir(parents=True, exist_ok=True)
    return out


def datasets_dir(module: str = "fe") -> Path:
    base = fe_root() / "output" / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def setup_notebook_path():
    """Notebook'tan import için sys.path ayarla."""
    import sys

    root = Path.cwd()
    if root.name == "notebooks":
        root = root.parent
    elif root.name != "eda" and (root / "eda").exists():
        root = root / "eda"
    elif root.name == "feature_engineering":
        pass
    elif (root / "feature_engineering").exists() and root.name != "eda":
        # proje kökünden FE notebook
        fe = root / "feature_engineering"
        if str(fe) not in sys.path:
            sys.path.insert(0, str(fe))
        if str(root / "eda") not in sys.path:
            sys.path.insert(0, str(root / "eda"))
        return root / "eda", fe
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root, None
