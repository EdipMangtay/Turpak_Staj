# Turpak – Staj Çalışmaları

Bu depo, **Turpak** stajım süresince geliştirdiğim çalışmaları, prototipleri ve tamamladığım taskları içerir.

**Stajyer:** Ali Edip Mangtay

---

## İçindekiler

| Bölüm | Açıklama |
|-------|----------|
| [strapping_table_model.py](strapping_table_model.py) | Yatay silindirik tank için seviye → hacim (cetvel) interpolasyon modeli |
| [wetstock_generator.py](wetstock_generator.py) | Sentetik wetstock veri ambarı üretici (8 tablo, 90 gün) |
| [eda/](eda/) | Keşifsel veri analizi (notebook'lar + EDA_Master) |
| [feature_engineering/](feature_engineering/) | ML feature engineering (Gün 8 notebook) |
| [data/](data/) | Sentetik operasyonel CSV'ler + `features.csv` |

> Tüm wetstock verisi **sentetik ve anonim** — gerçek istasyon verisi içermez.
> `data/ground_truth/` yalnızca ML değerlendirmesi içindir.

---

## Klasör yapısı

```
Staj/
├── data/                          # Sentetik CSV'ler (8 tablo + features.csv)
│   └── ground_truth/              # ML etiketleri
├── eda/                           # Keşifsel veri analizi
│   ├── notebooks/                 # 01–10 ayrı notebook'lar
│   ├── EDA_Master.ipynb           # Birleşik EDA
│   └── utils/
├── feature_engineering/
│   ├── notebooks/GUN08_feature_engineering.ipynb
│   └── fe_utils/
└── wetstock_generator.py
```

---

## Kurulum

```bash
pip install -r eda/requirements.txt
```

## Hızlı başlangıç

```bash
# Sentetik veri üret
python wetstock_generator.py

# EDA
cd eda && python build_master.py
jupyter notebook EDA_Master.ipynb

# Feature engineering (Gün 8)
cd ../feature_engineering && python build_gun8_notebook.py
jupyter notebook notebooks/GUN08_feature_engineering.ipynb
```

---

## Projeler

### 1. Strapping Table (Cetvel) Interpolasyon Modeli

Yatay silindirik akaryakıt tankında seviye–hacim ilişkisi doğrusal değildir. Script geometriden tam eğriyi hesaplar, cetvel tablosu oluşturur ve interpolasyon hatasını ölçer.

```bash
python strapping_table_model.py
```

### 2. Sentetik Wetstock Veri Ambarı

WSM 7-sekmeli arayüze uygun 8 operasyonel tablo: stations, tanks, mapping, transactions, ue1t_30min, inventory_30min, deliveries, daily.

### 3. EDA + Feature Engineering

Katman tutarlılığı doğrulaması, null analizi, anomali sinyalleri ve 6 feature grubu (`kk_oran`, sıcaklık, rolling MA, kümülatif eğim, manifold eş-tank).

Çıktı: `data/features.csv` (138K satır, 30 kolon)
