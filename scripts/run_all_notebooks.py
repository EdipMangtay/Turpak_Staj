#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projedeki tüm aktif Jupyter notebook'larını çalıştırır (çıktılar dosyaya yazılır)."""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TIMEOUT = 600

NOTEBOOKS = [
    # EDA — günlük notebook'lar (cwd = notebooks/)
    (PROJECT / "eda" / "notebooks" / "01_veri_semasi_ve_ozet.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "02_katman_tutarliligi.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "03_null_ve_eksik_veri.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "04_tek_tank_derinlemesine.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "05_gunluk_alarm_ve_fark.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "06_envanter_su_sicaklik.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "07_satis_ve_dolum.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "08_manifold_bolmeli.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "09_anomali_kesfi.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "notebooks" / "10_gun07_derin_eda.ipynb", PROJECT / "eda" / "notebooks"),
    (PROJECT / "eda" / "EDA_Master.ipynb", PROJECT / "eda"),
    # Feature engineering
    (PROJECT / "feature_engineering" / "notebooks" / "FE_Master.ipynb", PROJECT / "feature_engineering" / "notebooks"),
    (PROJECT / "feature_engineering" / "notebooks" / "GUN08_feature_engineering.ipynb", PROJECT / "feature_engineering" / "notebooks"),
]


def run_notebook(nb_path: Path, cwd: Path) -> bool:
    if not nb_path.exists():
        print(f"  ATLANDI (yok): {nb_path.relative_to(PROJECT)}")
        return True
    print(f"\n{'='*60}")
    print(f"Çalıştırılıyor: {nb_path.relative_to(PROJECT)}")
    print(f"{'='*60}")
    r = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", str(nb_path),
            f"--ExecutePreprocessor.timeout={TIMEOUT}",
            "--output", nb_path.name,
            "--output-dir", str(nb_path.parent),
        ],
        cwd=str(cwd),
    )
    if r.returncode != 0:
        print(f"  HATA: {nb_path.name}")
        return False
    print(f"  OK: {nb_path.name}")
    return True


def main():
    # Önce FE notebook'larını güncel şablonla yeniden oluştur
    build = PROJECT / "build_master.py"
    if build.exists():
        print("Notebook şablonları yenileniyor (build_master.py --fe)...")
        subprocess.run([sys.executable, str(build), "--fe"], check=True, cwd=str(PROJECT))

    failed = []
    for nb_path, cwd in NOTEBOOKS:
        if not run_notebook(nb_path, cwd):
            failed.append(str(nb_path.relative_to(PROJECT)))

    if failed:
        print("\nBaşarısız notebook'lar:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\nTamamlandı — {len(NOTEBOOKS)} notebook çalıştırıldı.")


if __name__ == "__main__":
    main()
