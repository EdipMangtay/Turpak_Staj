#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature engineering master notebook'unu çalıştırır."""
import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).parent
    project = root.parent
    nb = root / "notebooks" / "FE_Master.ipynb"
    if not nb.exists():
        subprocess.run([sys.executable, str(project / "build_master.py"), "--fe"], check=True)
    print(f"Çalıştırılıyor: {nb.name}")
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", str(nb), f"--ExecutePreprocessor.timeout=600",
         "--output", nb.name, "--output-dir", str(nb.parent)],
        cwd=str(root),
    )
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
