# Arşiv — Eski build scriptleri

Bu klasördeki dosyalar **notebook üreticilerinin eski sürümleri**.  
Günlük kullanım için proje kökündeki tek script yeterli:

```bash
python build_master.py
```

| Dosya | Eski işlev |
|-------|------------|
| `build_eda_master.py` | `eda/EDA_Master.ipynb` birleştirici (eda/ içindeydi) |
| `build_fe_master.py` | `feature_engineering/notebooks/FE_Master.ipynb` |
| `build_gun07_notebook.py` | `10_gun07_derin_eda.ipynb` üretici |
| `build_gun8_notebook.py` | `GUN08_feature_engineering.ipynb` üretici |
| `build_tam_rehber.py` | Eski tek dosyalık EDA rehberi |
| `create_notebooks.py` | 01–09 notebook'larını sıfırdan oluşturan generator |

Notebook'lar zaten `eda/notebooks/` ve `feature_engineering/notebooks/` altında mevcut; yeniden üretmek gerekirse bu scriptler referans olarak kalır.
