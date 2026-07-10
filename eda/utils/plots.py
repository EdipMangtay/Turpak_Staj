# -*- coding: utf-8 -*-
"""Grafik yardımcıları — birleşik görselleştirme çıktıları."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt

from .paths import figures_dir

# Aktif bölüm — master notebook setup'ta set edilir
_CURRENT_SECTION = "genel"


def set_section(section: str):
    """Sonraki save_fig çağrıları bu alt klasöre gider."""
    global _CURRENT_SECTION
    _CURRENT_SECTION = section


def setup_style():
    """Tutarlı grafik stili."""
    plt.rcParams.update({
        "figure.figsize": (12, 4),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def save_fig(
    fig=None,
    name: str = "plot",
    section: Optional[str] = None,
    module: str = "eda",
    dpi: int = 120,
    show: bool = True,
) -> Path:
    """Grafiği bölüm klasörüne kaydet ve isteğe bağlı göster."""
    sec = section or _CURRENT_SECTION
    out_dir = figures_dir(module=module, section=sec)
    path = out_dir / f"{name}.png"
    target = fig if fig is not None else plt.gcf()
    target.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return path


def save_or_show(fig=None, path: Optional[Union[str, Path]] = None, show: bool = True):
    """Geriye uyumluluk — path verilirse kaydet."""
    if path:
        target = fig if fig is not None else plt.gcf()
        target.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
