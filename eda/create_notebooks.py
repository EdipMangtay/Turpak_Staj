#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA Jupyter notebook'larını oluşturur."""

import json
from pathlib import Path

OUT = Path(__file__).parent


def nb(cells):
    return {
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


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


NOTEBOOKS = {
    "01_veri_semasi_ve_ozet.ipynb": [
        md("""# 01 — Veri Şeması ve Genel Özet

Bu notebook sentetik wetstock veri ambarının **ilk tanışma** adımıdır.

**Kapsam:**
- 8 operasyonel tablonun yüklenmesi
- Satır/kolon/null özeti
- İstasyon ve tank dağılımı
- Tarih aralığı ve granülarite

> `ground_truth/` klasörünü bu aşamada **kullanmıyoruz** — anomalileri kendimiz keşfedeceğiz."""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all, summary_table, DATA_DIR
from utils.plots import setup_style

setup_style()
sns.set_theme(style='whitegrid')
print('Veri klasörü:', DATA_DIR)
dfs = load_all()
list(dfs.keys())"""),
        code("""# Tablo özeti
ozet = summary_table(dfs)
display(ozet)"""),
        code("""# Her tablonun kolonları
for name, df in dfs.items():
    print('=' * 60)
    print(f'{name}: {df.shape[0]:,} satır x {df.shape[1]} kolon')
    print('Kolonlar:', list(df.columns))
    print(df.dtypes)
    print()"""),
        code("""# İstasyon × tank matrisi
stations = dfs['stations']
tanks = dfs['tanks']
print(stations.to_string(index=False))
print()
print('Tank sayısı istasyon başına:')
print(tanks.groupby('istasyon_kodu').size().to_string())"""),
        code("""# Ürün ve kapasite dağılımı
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tanks['akaryakit_turu'].value_counts().plot(kind='bar', ax=axes[0], title='Ürün dağılımı')
tanks.boxplot(column='kapasite', by='istasyon_kodu', ax=axes[1])
axes[1].set_title('Kapasite dağılımı (istasyon)')
plt.suptitle('')
plt.tight_layout()
plt.show()"""),
        code("""# Tarih aralığı
daily = dfs['daily']
tx = dfs['transactions']
print('Daily  :', daily.tarih.min(), '→', daily.tarih.max(), f'({daily.tarih.nunique()} gün)')
print('TX     :', tx.satis_zamani.min(), '→', tx.satis_zamani.max())
print('Beklenen daily satır (32 tank × 90 gün):', 32 * 90)
print('Gerçek daily satır:', len(daily))"""),
        code("""# Granülarite tablosu
gran = pd.DataFrame([
    ['transactions', 'tekil satış', 'değişken (~100+/gün/tank)'],
    ['ue1t_30min', '30 dakika', '48/gün/tank'],
    ['inventory_30min', '30 dakika', '48/gün/tank'],
    ['daily', 'günlük', '1/gün/tank'],
    ['deliveries', 'olay bazlı', 'dolum olduğunda'],
], columns=['tablo', 'granülarite', 'beklenen'])
display(gran)"""),
        md("""## Sonuç

- Veri **yıldız şeması**: `istasyon_kodu + tank_no` ile tüm tablolar birbirine bağlanır.
- Sonraki notebook: katman tutarlılığı doğrulaması (`02_katman_tutarliligi.ipynb`)."""),
    ],

    "02_katman_tutarliligi.ipynb": [
        md("""# 02 — Katman Tutarlılığı (Cross-Validation)

Gerçek WSM sisteminde farklı sekmeler (Satış, UE1T, Günlük) birbirini doğrular.

**Kontrol edilecekler:**
1. UE1T iç mutabakat denklemi
2. Dönem sürekliliği (son → sonraki baş)
3. transactions → ue1t (30 dk toplam)
4. ue1t → daily (günlük toplam)
5. deliveries → daily (dolum)"""),
        code("""import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all
from utils.validation import run_all_checks, check_ue1t_balance
from utils.plots import setup_style

setup_style()
dfs = load_all()
checks = run_all_checks(dfs)
checks['summary']"""),
        code("""# 1) UE1T mutabakat denklemi
ue1t_bal = checks['ue1t_balance']
print('Mutabakat residual istatistikleri:')
print(ue1t_bal['mutabakat_residual'].describe())
print('Max abs:', ue1t_bal['mutabakat_residual'].abs().max())

fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(ue1t_bal['mutabakat_residual'], bins=50, edgecolor='k', alpha=0.7)
ax.set_xlabel('Residual (lt)')
ax.set_title('UE1T mutabakat denklemi residual dağılımı')
plt.show()"""),
        code("""# 2) Süreklilik — kesinti dışı beklenen ~0
cont = checks['continuity']
big = cont[cont['sureklilik_fark'].abs() > 0.01]
print(f'Süreklilik |fark|>0.01 satır: {len(big)} (kesinti/boşluk normal)')
if len(big):
    display(big[['istasyon_kodu','tank_no','saat_1','onceki_son','donem_basi_stok','sureklilik_fark']].head(10))"""),
        code("""# 3) TX → UE1T
txu = checks['tx_to_ue1t']
mismatch = txu[txu['tx_ue1t_fark'].abs() > 0.01]
print(f'TX vs UE1T uyuşmayan dönem: {len(mismatch)}')
print(f'Unmapped satış litre toplamı: {dfs["transactions"].loc[dfs["transactions"].tank_no.isna(), "litre"].sum():.1f}')

fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(txu['tx_ue1t_fark'], bins=80, edgecolor='k', alpha=0.7)
ax.set_xlabel('pompa_satis - tx_toplam (lt)')
ax.set_title('TX → UE1T fark dağılımı')
plt.show()"""),
        code("""# 4) UE1T → Daily
ud = checks['ue1t_to_daily']
for col in ['satis_fark','dolum_fark','acilis_fark','kapanis_fark']:
    print(f'{col}: max abs = {ud[col].abs().max():.6f}')

alarm_gun = ud[ud['satis_fark'].abs() > 0.01]
if len(alarm_gun):
    display(alarm_gun.head())"""),
        code("""# 5) Deliveries → Daily
dd = checks['deliveries_to_daily']
bad = dd[dd['delivery_fark'].abs() > 0.01]
print(f'Delivery vs daily fark >0.01 gün: {len(bad)}')
print('(Gün sonu dolum kayması olabilir — EDA bulgusu)')
if len(bad):
    display(bad[['tarih','istasyon_kodu','tank_no','dolum','dolum_net','delivery_fark']].head(10))"""),
        code("""# Tek tank, tek gün derin doğrulama
IST, TANK, GUN = 'IST_001', 1, pd.Timestamp('2026-01-01')
u = dfs['ue1t_30min'][(dfs['ue1t_30min'].istasyon_kodu==IST)&(dfs['ue1t_30min'].tank_no==TANK)]
u = u[u.saat_1.dt.normalize()==GUN]
t = dfs['transactions'][(dfs['transactions'].istasyon_kodu==IST)&(dfs['transactions'].tank_no==TANK)]
t = t[t.satis_zamani.dt.normalize()==GUN]
d = dfs['daily'][(dfs['daily'].istasyon_kodu==IST)&(dfs['daily'].tank_no==TANK)&(dfs['daily'].tarih==GUN)]

print('=== IST_001 Tank 1 — 2026-01-01 ===')
print(f'daily.satis      = {d.satis.iloc[0]:.2f}')
print(f'ue1t toplam      = {u.pompa_satis.sum():.2f}')
print(f'tx toplam        = {t.litre.sum():.2f}')
print(f'daily.acilis     = {d.acilis.iloc[0]:.2f}  | ue1t ilk = {u.donem_basi_stok.iloc[0]:.2f}')
print(f'daily.kapanis    = {d.kapanis.iloc[0]:.2f}  | ue1t son = {u.donem_sonu_stok.iloc[-1]:.2f}')"""),
        md("""## Sonuç

Katmanlar tutarlı. TX↔UE1T farkları çoğunlukla **unmapped satış** (`tank_no` null) kaynaklı.

Sonraki: null analizi (`03_null_ve_eksik_veri.ipynb`)."""),
    ],

    "03_null_ve_eksik_veri.ipynb": [
        md("""# 03 — Null ve Eksik Veri Analizi

Sentetik veride bilinçli null'lar var — EDA'da **neden eksik?** sorusunu cevaplayacağız."""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all
from utils.plots import setup_style

setup_style()
dfs = load_all()"""),
        code("""# Genel null matrisi
null_rows = []
for name, df in dfs.items():
    for col in df.columns:
        n = df[col].isna().sum()
        if n:
            null_rows.append({'tablo': name, 'kolon': col, 'null_sayisi': n,
                              'oran_pct': round(100*n/len(df), 3)})
null_df = pd.DataFrame(null_rows).sort_values('null_sayisi', ascending=False)
display(null_df)"""),
        code("""# Heatmap — tablo × kolon null oranı
pivot = null_df.pivot_table(index='tablo', columns='kolon', values='oran_pct', fill_value=0)
plt.figure(figsize=(14, 5))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Null oranı (%) — tablo × kolon')
plt.tight_layout()
plt.show()"""),
        code("""# transactions.tank_no null — istasyon kırılımı
tx = dfs['transactions']
unmapped = tx[tx['tank_no'].isna()]
print('Unmapped satış:', len(unmapped), f'({100*len(unmapped)/len(tx):.2f}%)')
print(unmapped.groupby('istasyon_kodu').size().sort_values(ascending=False))"""),
        code("""# sicaklik null — tank×gün kümesi mi?
ue1t = dfs['ue1t_30min']
inv = dfs['inventory_30min']
for name, df, col in [('ue1t','ue1t_30min','sicaklik'), ('inventory','inventory_30min','sicaklik')]:
    d = dfs[name.replace('inventory','inventory_30min') if name=='inventory' else 'ue1t_30min']
    sn = d[d['sicaklik'].isna()].copy()
    if len(sn):
        sn['gun'] = sn['saat_1' if 'saat_1' in sn.columns else 'envanter_tarihi'].dt.date
        print(f'\\n{name} sicaklik null: {len(sn)} satır')
        print('Tank-gün kümesi (ilk 10):')
        print(sn.groupby(['istasyon_kodu','tank_no','gun']).size().head(10))"""),
        code("""# merkeze_gelis gecikmesi — router arızası imzası
inv = dfs['inventory_30min'].copy()
inv['gecikme_dk'] = (inv['merkeze_gelis_tarihi'] - inv['envanter_tarihi']).dt.total_seconds() / 60
print('Gecikme istatistikleri (dk):')
print(inv['gecikme_dk'].describe())

# En yüksek gecikmeli istasyon-gün
inv['tarih'] = inv['envanter_tarihi'].dt.normalize()
lag_day = inv.groupby(['istasyon_kodu','tarih'])['gecikme_dk'].median().reset_index()
top = lag_day.nlargest(5, 'gecikme_dk')
print('\\nEn yüksek medyan gecikme (istasyon-gün):')
display(top)"""),
        code("""fig, ax = plt.subplots(figsize=(12, 4))
inv.boxplot(column='gecikme_dk', by='istasyon_kodu', ax=ax)
ax.set_title('Merkeze geliş gecikmesi (dk) — istasyon')
ax.set_xlabel('İstasyon')
plt.suptitle('')
plt.show()"""),
        code("""# Eksik 30dk satırları — beklenen vs gerçek
tanks = dfs['tanks']
ue1t = dfs['ue1t_30min']
n_gun = dfs['daily'].tarih.nunique()
beklenen = n_gun * 48
cnt = ue1t.groupby(['istasyon_kodu','tank_no']).size().reset_index(name='satir')
tank_list = tanks[['istasyon_kodu','tank_no']]
cnt = tank_list.merge(cnt, on=['istasyon_kodu','tank_no'], how='left').fillna(0)
cnt['eksik'] = beklenen - cnt['satir']
eksik = cnt[cnt['eksik'] > 0]
print(f'Beklenen dönem/tank: {beklenen}')
print(f'Eksik satırı olan tank sayısı: {len(eksik)}')
if len(eksik):
    display(eksik.sort_values('eksik', ascending=False).head(10))"""),
        md("""## Doldurma stratejisi notları (feature engineering için)

| Kolon | Öneri |
|---|---|
| `sicaklik` null | Aynı tank-gün interpolasyon veya forward fill |
| `tank_no` null | mapping join veya ayrı "unmapped" grubu |
| `birim_fiyat` null | tutar/litre veya gün medyanı |
| `merkeze_gelis_tarihi` null | Gecikme feature olarak bırak |

Sonraki: tek tank derinlemesine analiz (`04_tek_tank_derinlemesine.ipynb`)."""),
    ],

    "04_tek_tank_derinlemesine.ipynb": [
        md("""# 04 — Tek Tank Derinlemesine Analiz

**IST_001, Tank 1** (Motorin, 23000 lt, manifoldlu) üzerinde gerçek sistemdeki
"istasyon seç → tank detayları" akışını simüle ediyoruz."""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all, filter_tank, merge_ue1t_inventory
from utils.plots import setup_style

setup_style()
dfs = load_all()

IST, TANK = 'IST_001', 1
tank_info = filter_tank(dfs['tanks'], IST, TANK)
daily = filter_tank(dfs['daily'], IST, TANK).sort_values('tarih')
ue1t = filter_tank(dfs['ue1t_30min'], IST, TANK).sort_values('saat_1')
tx = filter_tank(dfs['transactions'], IST, TANK).sort_values('satis_zamani')
inv = filter_tank(dfs['inventory_30min'], IST, TANK).sort_values('envanter_tarihi')
deliv = filter_tank(dfs['deliveries'], IST, TANK)

print('Tank bilgisi:')
display(tank_info.T)
print(f'\\nSatır sayıları: daily={len(daily)}, ue1t={len(ue1t)}, tx={len(tx)}, inv={len(inv)}, deliv={len(deliv)}')"""),
        code("""# Günlük zaman serisi — 4 panel
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
daily.plot(x='tarih', y='satis', ax=axes[0], legend=False, color='steelblue')
axes[0].set_ylabel('Satış (lt)')
daily.plot(x='tarih', y='fark', ax=axes[1], legend=False, color='coral')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('Fark (lt)')
daily.plot(x='tarih', y='kapanis', ax=axes[2], legend=False, color='green')
axes[2].axhline(tank_info.kapasite.iloc[0]*0.04, color='r', ls='--', label='dead stock ~4%')
axes[2].set_ylabel('Kapanış stok')
daily.plot(x='tarih', y='alarm', ax=axes[3], legend=False, drawstyle='steps-post', color='red')
axes[3].set_ylabel('Alarm (0/1)')
axes[3].set_xlabel('Tarih')
fig.suptitle(f'{IST} Tank {TANK} — 90 günlük özet', fontsize=14)
plt.tight_layout()
plt.show()"""),
        code("""# Tek gün 30 dk detay — en yüksek |fark| günü
gun = daily.loc[daily['fark'].abs().idxmax(), 'tarih']
u_gun = ue1t[ue1t.saat_1.dt.normalize() == gun]
print(f'Seçilen gün: {gun.date()} | günlük fark: {daily[daily.tarih==gun].fark.values[0]:.2f} lt')

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
u_gun.plot(x='saat_1', y='donem_sonu_stok', ax=axes[0], legend=False)
axes[0].set_ylabel('Stok (lt)')
u_gun.plot(x='saat_1', y='pompa_satis', ax=axes[1], legend=False, color='orange')
axes[1].set_ylabel('30dk satış')
u_gun.plot(x='saat_1', y='kayip_kazanc', ax=axes[2], legend=False, color='red')
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_ylabel('Kayıp/Kazanç')
axes[2].set_xlabel('Saat')
fig.suptitle(f'{IST} T{TANK} — {gun.date()} UE1T detay')
plt.tight_layout()
plt.show()"""),
        code("""# Saatlik satış profili (tüm dönem)
tx2 = tx.copy()
tx2['saat'] = tx2['satis_zamani'].dt.hour + tx2['satis_zamani'].dt.minute/60
profil = tx2.groupby(tx2['satis_zamani'].dt.floor('h'))['litre'].sum()
profil.index = profil.index.hour

hourly = tx2.groupby(tx2['satis_zamani'].dt.hour)['litre'].sum()
fig, ax = plt.subplots(figsize=(12, 4))
hourly.plot(kind='bar', ax=ax, color='steelblue', edgecolor='k')
ax.set_xlabel('Saat')
ax.set_ylabel('Toplam litre')
ax.set_title(f'{IST} T{TANK} — saatlik satış profili (90 gün)')
plt.tight_layout()
plt.show()"""),
        code("""# UE1T + Envanter join
merged = merge_ue1t_inventory(ue1t, inv)
print('Join sonrası satır:', len(merged), '| null sicaklik:', merged['sicaklik_inv'].isna().sum() if 'sicaklik_inv' in merged.columns else merged.filter(like='sicaklik').isna().sum().sum())

fig, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(merged['saat_1'], merged['donem_sonu_stok'], label='Stok (UE1T)', color='steelblue')
ax1.set_ylabel('Stok (lt)')
ax2 = ax1.twinx()
ax2.plot(merged['saat_1'], merged['sicaklik_ue1t'], label='Sıcaklık', color='orange', alpha=0.7)
ax2.set_ylabel('Sıcaklık (°C)')
ax1.set_title(f'{IST} T{TANK} — stok ve sıcaklık')
fig.tight_layout()
plt.show()"""),
        code("""# Dolum olayları
if len(deliv):
    display(deliv[['dolum_baslangic','dolum_net','dolum_oncesi_hacim','dolum_sonrasi_hacim','sicaklik']].head(10))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(deliv['dolum_baslangic'], deliv['dolum_net'], s=60, alpha=0.7)
    ax.set_xlabel('Dolum zamanı')
    ax.set_ylabel('Dolum net (lt)')
    ax.set_title(f'{IST} T{TANK} — dolum olayları')
    plt.show()"""),
        md("""## Sonuç

Tek tank üzerinde günlük özet → 30 dk detay → tekil satış → envanter akışını gördük.

Sonraki: tüm tanklar için günlük alarm analizi (`05_gunluk_alarm_ve_fark.ipynb`)."""),
    ],

    "05_gunluk_alarm_ve_fark.ipynb": [
        md("""# 05 — Günlük Mutabakat, Fark ve Alarm Analizi

WSM Günlük Analiz tablosu (`daily.csv`) — SEL eşiği ve alarm mantığı."""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all
from utils.plots import setup_style

setup_style()
daily = load_all()['daily']
daily['fark_abs'] = daily['fark'].abs()
daily['sel_asildi'] = daily['fark_abs'] > daily['sel']"""),
        code("""print('Alarm oranı:', daily['alarm'].mean())
print('SEL aşım oranı (manuel kontrol):', daily['sel_asildi'].mean())
print('Alarm ile SEL tutarlı mı:', (daily['alarm'] == daily['sel_asildi'].astype(int)).mean())"""),
        code("""# Fark dağılımı
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
daily['fark'].hist(bins=80, ax=axes[0], edgecolor='k', alpha=0.7)
axes[0].set_xlabel('Fark (lt)'); axes[0].set_title('Günlük fark dağılımı')
daily['oran'].hist(bins=80, ax=axes[1], edgecolor='k', alpha=0.7, color='coral')
axes[1].set_xlabel('Oran (%)'); axes[1].set_title('Günlük oran dağılımı')
plt.tight_layout()
plt.show()"""),
        code("""# Alarm sayısı — istasyon × tank
alarm_cnt = daily.groupby(['istasyon_kodu','tank_no'])['alarm'].sum().reset_index()
alarm_cnt = alarm_cnt.sort_values('alarm', ascending=False)
print('En çok alarm üreten tanklar:')
display(alarm_cnt.head(15))

plt.figure(figsize=(10, 5))
top = alarm_cnt.head(12)
labels = top['istasyon_kodu'] + '-T' + top['tank_no'].astype(str)
plt.barh(labels, top['alarm'], color='crimson', edgecolor='k')
plt.xlabel('Alarm gün sayısı')
plt.title('En çok alarm — istasyon/tank')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""# Fark vs satış scatter
plt.figure(figsize=(8, 6))
colors = daily['alarm'].map({0: 'steelblue', 1: 'crimson'})
plt.scatter(daily['satis'], daily['fark'], c=colors, alpha=0.4, s=15)
plt.axhline(0, color='k', lw=0.5)
plt.xlabel('Günlük satış (lt)')
plt.ylabel('Fark (lt)')
plt.title('Fark vs Satış (kırmızı=alarm)')
plt.tight_layout()
plt.show()"""),
        code("""# Mutabakat denklemi kontrolü: fark = satis - azalma
daily['fark_hesap'] = daily['satis'] - daily['azalma_miktari']
daily['fark_residual'] = daily['fark'] - daily['fark_hesap']
print('Fark denklemi max residual:', daily['fark_residual'].abs().max())"""),
        md("""Sonraki: envanter, su, sıcaklık (`06_envanter_su_sicaklik.ipynb`)."""),
    ],

    "06_envanter_su_sicaklik.ipynb": [
        md("""# 06 — Envanter, Su Faktörü ve Sıcaklık

`inventory_30min.csv` — brüt/net hacim, su seviyesi, merkeze geliş gecikmesi."""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all
from utils.plots import setup_style

setup_style()
inv = load_all()['inventory_30min']
inv['brut_net_fark'] = inv['urun_miktari_brut'] - inv['urun_miktari_net']
inv['gecikme_dk'] = (inv['merkeze_gelis_tarihi'] - inv['envanter_tarihi']).dt.total_seconds()/60"""),
        code("""# Brüt vs Net
fig, ax = plt.subplots(figsize=(8, 6))
sample = inv.dropna(subset=['urun_miktari_net']).sample(min(5000, len(inv)), random_state=42)
ax.scatter(sample['urun_miktari_brut'], sample['urun_miktari_net'], alpha=0.2, s=5)
ax.plot([0, sample['urun_miktari_brut'].max()], [0, sample['urun_miktari_brut'].max()], 'r--', lw=1)
ax.set_xlabel('Brüt (lt)'); ax.set_ylabel('Net (lt)')
ax.set_title('Brüt vs Net hacim')
plt.show()

print('Brüt-Net fark istatistikleri:')
print(inv['brut_net_fark'].describe())"""),
        code("""# Su seviyesi — sıfır olmayan tank-günler
su_pos = inv[inv['su_seviyesi_cm'].fillna(0) > 0.5]
print('Su > 0.5 cm satır:', len(su_pos))
if len(su_pos):
    su_tank = su_pos.groupby(['istasyon_kodu','tank_no']).size().sort_values(ascending=False)
    print('En çok su kaydı olan tanklar:')
    print(su_tank.head(10))"""),
        code("""# Su artış hızı — su faktörü adayı
inv2 = inv.sort_values(['istasyon_kodu','tank_no','envanter_tarihi']).copy()
inv2['su_diff'] = inv2.groupby(['istasyon_kodu','tank_no'])['su_seviyesi_cm'].diff()
spike = inv2[inv2['su_diff'] > 0.05]
print('Su sıçrama (>0.05 cm/30dk) satır:', len(spike))
if len(spike):
    display(spike.nlargest(10, 'su_diff')[['envanter_tarihi','istasyon_kodu','tank_no','su_seviyesi_cm','su_diff']])"""),
        code("""# Sıcaklık dağılımı
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
inv['sicaklik'].dropna().hist(bins=50, ax=axes[0], edgecolor='k')
axes[0].set_title('Sıcaklık dağılımı')
inv.boxplot(column='sicaklik', by='istasyon_kodu', ax=axes[1])
axes[1].set_title('Sıcaklık — istasyon')
plt.suptitle('')
plt.tight_layout()
plt.show()"""),
        code("""# Seviye (cm) vs hacim (lt) — strapping ilişkisi
sample = inv.dropna(subset=['urun_seviyesi_cm','urun_miktari_brut']).sample(min(3000,len(inv)), random_state=1)
plt.figure(figsize=(8, 5))
for (ist, tn), g in sample.groupby(['istasyon_kodu','tank_no']):
    if len(g) > 20:
        plt.scatter(g['urun_seviyesi_cm'], g['urun_miktari_brut'], alpha=0.3, s=5, label=f'{ist}-T{tn}')
plt.xlabel('Seviye (cm)'); plt.ylabel('Brüt hacim (lt)')
plt.title('Seviye–Hacim (cetvel) ilişkisi')
plt.legend(bbox_to_anchor=(1.05,1), fontsize=7, ncol=2)
plt.tight_layout()
plt.show()"""),
        md("""Sonraki: satış ve dolum (`07_satis_ve_dolum.ipynb`)."""),
    ],

    "07_satis_ve_dolum.ipynb": [
        md("""# 07 — Satış ve Dolum Analizi

`transactions.csv` ve `deliveries.csv` — pompa decimal, test satışı, hayali/algılanmayan dolum sinyalleri."""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all
from utils.plots import setup_style

setup_style()
dfs = load_all()
tx = dfs['transactions']
deliv = dfs['deliveries']"""),
        code("""# Satış istatistikleri
print('Toplam işlem:', len(tx))
print('Test satışı:', tx['test_satisi'].sum())
print('Satış tipi:', tx['satis_tipi'].value_counts().to_dict())
print('Litre istatistikleri:')
print(tx['litre'].describe())"""),
        code("""# Pompa decimal adayı: litre vs tutar/birim_fiyat
tx_ok = tx.dropna(subset=['litre','tutar','birim_fiyat'])
tx_ok = tx_ok[tx_ok['birim_fiyat'] > 0]
tx_ok['litre_hesap'] = tx_ok['tutar'] / tx_ok['birim_fiyat']
tx_ok['litre_oran'] = tx_ok['litre'] / tx_ok['litre_hesap']

# 10x sapma
decimal = tx_ok[(tx_ok['litre_oran'] > 5) | (tx_ok['litre_oran'] < 0.2)]
print('Decimal şüpheli işlem:', len(decimal))
if len(decimal):
    display(decimal.groupby(['istasyon_kodu','tank_no']).size().sort_values(ascending=False).head())"""),
        code("""# İşlem büyüklüğü dağılımı
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tx['litre'].hist(bins=80, ax=axes[0], edgecolor='k', alpha=0.7)
axes[0].set_xlabel('Litre'); axes[0].set_title('Tekil satış litre dağılımı')
tx.groupby('istasyon_kodu')['litre'].sum().plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='k')
axes[1].set_title('Toplam satış — istasyon')
plt.tight_layout()
plt.show()"""),
        code("""# Dolum analizi
print('Toplam dolum kaydı:', len(deliv))
print('Dolum net istatistikleri:')
print(deliv['dolum_net'].describe())

deliv['tarih'] = deliv['dolum_baslangic'].dt.normalize()
dolum_gun = deliv.groupby(['istasyon_kodu','tank_no','tarih'])['dolum_net'].sum().reset_index()
print('\\nEn büyük dolumlar:')
display(dolum_gun.nlargest(10, 'dolum_net'))"""),
        code("""# Dolum sonrası envanter artışı vs kayıtlı dolum — ue1t join
ue1t = dfs['ue1t_30min']
dolum_donem = ue1t[ue1t['tanka_dolum'] > 100][['saat_1','istasyon_kodu','tank_no','tanka_dolum','kayip_kazanc']]
print('UE1T dolumlu dönem (>100 lt):', len(dolum_donem))
display(dolum_donem.nlargest(10, 'tanka_dolum'))"""),
        md("""Sonraki: manifold ve bölmeli tanklar (`08_manifold_bolmeli.ipynb`)."""),
    ],

    "08_manifold_bolmeli.ipynb": [
        md("""# 08 — Manifold ve Bölmeli Tank Analizi

Manifold: check-valve kaçırması → bir tankta kayıp, eşleşen tankta kazanç.
Bölmeli: dolumda sac baskılanması → karşılıklı geçici hareket."""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all
from utils.plots import setup_style

setup_style()
dfs = load_all()
tanks = dfs['tanks']
ue1t = dfs['ue1t_30min']

manifold = tanks[tanks['is_manifold']==1]
bolmeli = tanks[tanks['bolmeli']==1]
print('Manifoldlu tanklar:', len(manifold))
display(manifold[['istasyon_kodu','tank_no','manifold_grup_no','akaryakit_turu']])
print('\\nBölmeli tanklar:', len(bolmeli))
display(bolmeli[['istasyon_kodu','tank_no','bolme_grup_no']])"""),
        code("""# Manifold çiftleri — aynı saatte ters yönlü kayıp/kazanç
pairs = []
for (ist, grp), g in manifold.groupby(['istasyon_kodu','manifold_grup_no']):
    tn = sorted(g['tank_no'].tolist())
    if len(tn) == 2:
        pairs.append((ist, tn[0], tn[1]))

print('Manifold çift sayısı:', len(pairs))
corrs = []
for ist, a, b in pairs[:6]:
    ua = ue1t[(ue1t.istasyon_kodu==ist)&(ue1t.tank_no==a)].set_index('saat_1')['kayip_kazanc']
    ub = ue1t[(ue1t.istasyon_kodu==ist)&(ue1t.tank_no==b)].set_index('saat_1')['kayip_kazanc']
    m = pd.concat([ua, ub], axis=1, keys=['a','b']).dropna()
    if len(m) > 50:
        corr = m['a'].corr(m['b'])
        corrs.append({'ist': ist, 'tank_a': a, 'tank_b': b, 'corr': corr})
        print(f'{ist} T{a}-T{b}: kayip_kazanc korelasyon = {corr:.3f}')

pd.DataFrame(corrs)"""),
        code("""# Örnek manifold çifti grafiği
if pairs:
    ist, a, b = pairs[0]
    ua = ue1t[(ue1t.istasyon_kodu==ist)&(ue1t.tank_no==a)].sort_values('saat_1')
    ub = ue1t[(ue1t.istasyon_kodu==ist)&(ue1t.tank_no==b)].sort_values('saat_1')
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ua['saat_1'], ua['kayip_kazanc'].cumsum(), label=f'T{a}')
    ax.plot(ub['saat_1'], ub['kayip_kazanc'].cumsum(), label=f'T{b}')
    ax.set_ylabel('Kümülatif kayıp/kazanç')
    ax.legend()
    ax.set_title(f'{ist} Manifold çift T{a}-T{b}')
    plt.tight_layout()
    plt.show()"""),
        code("""# Bölmeli çift IST_003 T2-T3
ba, bb = 2, 3
ist = 'IST_003'
ua = ue1t[(ue1t.istasyon_kodu==ist)&(ue1t.tank_no==ba)].sort_values('saat_1')
ub = ue1t[(ue1t.istasyon_kodu==ist)&(ue1t.tank_no==bb)].sort_values('saat_1')
m = ua.merge(ub, on='saat_1', suffixes=('_a','_b'))
m['toplam_kk'] = m['kayip_kazanc_a'] + m['kayip_kazanc_b']

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
m.plot(x='saat_1', y=['kayip_kazanc_a','kayip_kazanc_b'], ax=axes[0])
axes[0].set_ylabel('30dk kayıp/kazanç')
axes[0].set_title(f'{ist} Bölmeli T{ba}-T{bb}')
m.plot(x='saat_1', y='toplam_kk', ax=axes[1], color='purple')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('Toplam (a+b)')
plt.tight_layout()
plt.show()"""),
        md("""Sonraki: anomali keşfi (`09_anomali_kesfi.ipynb`)."""),
    ],

    "09_anomali_kesfi.ipynb": [
        md("""# 09 — Anomali Sinyali Keşfi (Ground Truth Kullanmadan)

Kural tabanlı ve istatistiksel sinyaller — ML öncesi feature adayları.

> ML aşamasında `data/ground_truth/labels_30min.csv` ile karşılaştırarak
> precision/recall ölçebilirsin. **Bu notebook'ta açma.**"""),
        code("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
elif ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all, merge_ue1t_inventory
from utils.plots import setup_style

setup_style()
dfs = load_all()
ue1t = dfs['ue1t_30min'].copy()
inv = dfs['inventory_30min']"""),
        code("""# Feature adayları — 30dk bazında
ue1t['satisiz'] = (ue1t['pompa_satis'] == 0).astype(int)
ue1t['kk_oran'] = np.where(ue1t['pompa_satis'] > 1,
                           ue1t['kayip_kazanc'] / ue1t['pompa_satis'], np.nan)
ue1t['saat'] = ue1t['saat_1'].dt.hour
ue1t['gece'] = ue1t['saat'].between(0, 5).astype(int)

print('Feature özet:')
print(ue1t[['kayip_kazanc','kk_oran','satisiz','gece']].describe())"""),
        code("""# Sinyal 1: Satışsız kayıp (statik sızıntı adayı)
satisiz_kayip = ue1t[(ue1t['pompa_satis']==0) & (ue1t['kayip_kazanc'] < -5)]
print('Satışsız kayıp >5 lt dönem:', len(satisiz_kayip))
top = satisiz_kayip.nsmallest(10, 'kayip_kazanc')
display(top[['saat_1','istasyon_kodu','tank_no','kayip_kazanc','donem_basi_stok','donem_sonu_stok']])"""),
        code("""# Sinyal 2: Gece satışsız düşüş (pompacı manipülasyonu adayı)
gece = ue1t[(ue1t['gece']==1) & (ue1t['pompa_satis']==0) & (ue1t['kayip_kazanc'] < -20)]
print('Gece satışsız düşüş >20 lt:', len(gece))
if len(gece):
    display(gece.nsmallest(5,'kayip_kazanc')[['saat_1','istasyon_kodu','tank_no','kayip_kazanc']])"""),
        code("""# Sinyal 3: Yüksek kk_oran (dinamik sızıntı / decimal adayı)
high = ue1t[(ue1t['pompa_satis'] > 50) & (ue1t['kk_oran'].abs() > 0.05)]
print('|kk_oran| > 5% ve satış >50 lt:', len(high))
display(high.nlargest(10, 'kk_oran', keep='all')[['saat_1','istasyon_kodu','tank_no','pompa_satis','kayip_kazanc','kk_oran']].head(10))"""),
        code("""# Sinyal 4: Seviye donması (şamandıra adayı) — stok değişmeden satış
ue1t_sorted = ue1t.sort_values(['istasyon_kodu','tank_no','saat_1'])
ue1t_sorted['stok_diff'] = ue1t_sorted.groupby(['istasyon_kodu','tank_no'])['donem_sonu_stok'].diff()
stuck = ue1t_sorted[(ue1t_sorted['pompa_satis'] > 30) & (ue1t_sorted['stok_diff'].abs() < 0.01)]
print('Satış var ama stok değişmedi:', len(stuck))
if len(stuck):
    display(stuck.head(8)[['saat_1','istasyon_kodu','tank_no','pompa_satis','donem_sonu_stok','kayip_kazanc']])"""),
        code("""# Sinyal 5: Su sıçraması
inv2 = inv.sort_values(['istasyon_kodu','tank_no','envanter_tarihi']).copy()
inv2['su_diff'] = inv2.groupby(['istasyon_kodu','tank_no'])['su_seviyesi_cm'].diff()
su = inv2[inv2['su_diff'] > 0.05]
print('Su sıçrama:', len(su))"""),
        code("""# Günlük alarm günlerinde 30dk profili
daily = dfs['daily']
alarm_gun = daily[daily['alarm']==1].nlargest(1, 'fark')
if len(alarm_gun):
    row = alarm_gun.iloc[0]
    u = ue1t[(ue1t.istasyon_kodu==row.istasyon_kodu)&(ue1t.tank_no==row.tank_no)
             &(ue1t.saat_1.dt.normalize()==row.tarih)]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(u)), u['kayip_kazanc'], color='crimson', edgecolor='k')
    ax.axhline(0, color='k')
    ax.set_xlabel('30dk dönem')
    ax.set_ylabel('Kayıp/Kazanç')
    ax.set_title(f"Alarm günü {row.istasyon_kodu} T{row.tank_no} {row.tarih.date()} — fark={row.fark:.1f}")
    plt.tight_layout()
    plt.show()"""),
        md("""## Sonuç ve sonraki adımlar

**Keşfedilen sinyaller → feature engineering:**
- `satisiz_kayip`, `kk_oran`, `gece_satisiz_dusus`
- `tx_ue1t_fark`, `su_diff`, `gecikme_dk`
- Manifold çift korelasyonu

**ML aşaması:**
1. Feature'ları birleştir
2. Zaman bazlı train/test split (son 20 gün test)
3. `ground_truth/labels_30min.csv` ile karşılaştır
4. Baseline: SEL alarm vs CatBoost/IsolationForest"""),
    ],
}


def main():
    nb_dir = OUT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    for fname, cells in NOTEBOOKS.items():
        path = nb_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb(cells), f, ensure_ascii=False, indent=1)
        print("Created", path)


if __name__ == "__main__":
    main()
