#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tüm EDA notebook'larını sırayla çalıştırır."""
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "01_veri_semasi_ve_ozet.ipynb",
    "02_katman_tutarliligi.ipynb",
    "03_null_ve_eksik_veri.ipynb",
    "04_tek_tank_derinlemesine.ipynb",
    "05_gunluk_alarm_ve_fark.ipynb",
    "06_envanter_su_sicaklik.ipynb",
    "07_satis_ve_dolum.ipynb",
    "08_manifold_bolmeli.ipynb",
    "09_anomali_kesfi.ipynb",
    "10_gun07_derin_eda.ipynb",
]

def main():
    root = Path(__file__).parent
    nb_dir = root / "notebooks"
    for nb in NOTEBOOKS:
        path = nb_dir / nb
        print(f"\n{'='*60}\nÇalıştırılıyor: {nb}\n{'='*60}")
        r = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", str(path), f"--ExecutePreprocessor.timeout=300",
             "--output", path.name, "--output-dir", str(path.parent)],
            cwd=str(nb_dir),
        )
        if r.returncode != 0:
            print(f"HATA: {nb}")
            sys.exit(r.returncode)

    # Master notebook
    master = root / "EDA_Master.ipynb"
    if master.exists():
        print(f"\n{'='*60}\nÇalıştırılıyor: EDA_Master.ipynb\n{'='*60}")
        r = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", str(master), f"--ExecutePreprocessor.timeout=600",
             "--output", master.name, "--output-dir", str(master.parent)],
            cwd=str(nb_dir),
        )
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\nTüm notebook'lar tamamlandı.")

if __name__ == "__main__":
    main()
