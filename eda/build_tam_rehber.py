#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tek dosyalık EDA rehber notebook'unu oluşturur."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "EDA_Tam_Rehber.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "source": s, "outputs": [], "execution_count": None}


cells = [
md("""# Keşifsel Veri Analizi (EDA) — Tam Rehber

Bu notebook **tek dosyada**, sıfırdan başlayarak sentetik wetstock veri ambarını anlaman için hazırlandı.

**Sıra:** Basit → Orta → İleri (ilişkili tablolar, join, anomali sinyalleri)

**Klasör yapısı:**
```
Staj/
├── data/          ← ham CSV'ler (8 tablo)
└── eda/           ← bu notebook burada
```

> `data/ground_truth/` klasörünü **açma** — anomalileri kendin keşfedeceksin; ML aşamasında kullanırsın.

---"""),

code("""# ============================================================
# BÖLÜM 0 — Kurulum ve veri yükleme
# ============================================================
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Notebook eda/ klasöründen çalıştırılmalı
ROOT = Path.cwd()
if ROOT.name != 'eda':
    ROOT = ROOT / 'eda'
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_all, filter_tank, DATA_DIR, summary_table
from utils.validation import run_all_checks
from utils.plots import setup_style

setup_style()
sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

print('Veri klasörü:', DATA_DIR)
print('Klasör mevcut mu:', DATA_DIR.exists())"""),

code("""# 8 operasyonel tabloyu yükle
dfs = load_all()
print('Yüklenen tablolar:', list(dfs.keys()))"""),

md("""---
## BÖLÜM 1 — Veriyi tanı (BASİT)

Önce her tabloyu **tek başına** incele. Gerçek WSM sistemindeki sekmelere karşılık gelir."""),

code("""# 1.1 Tablo özeti — kaç satır, kaç kolon, null var mı?
ozet = summary_table(dfs)
display(ozet)"""),

code("""# 1.2 Her tablonun kolonları (ilk bakış)
for name, df in dfs.items():
    print('=' * 65)
    print(f'📁 {name.upper():20} → {len(df):,} satır × {df.shape[1]} kolon')
    print('   Kolonlar:', list(df.columns))
    print()"""),

code("""# 1.3 İstasyon ve tank sayıları
display(dfs['stations'])
print()
print('Tank / istasyon:')
print(dfs['tanks'].groupby('istasyon_kodu').size().to_string())"""),

code("""# 1.4 Tarih aralığı
daily = dfs['daily']
tx = dfs['transactions']
print('Günlük veri  :', daily.tarih.min().date(), '→', daily.tarih.max().date())
print('Gün sayısı   :', daily.tarih.nunique())
print('Satış verisi :', tx.satis_zamani.min(), '→', tx.satis_zamani.max())
print()
print('Granülarite (en ince → en kalın):')
print('  transactions     → tekil satış (en ince)')
print('  ue1t_30min       → 30 dakika')
print('  inventory_30min  → 30 dakika')
print('  daily            → günlük özet (en kalın)')"""),

code("""# 1.5 İlk 5 satır — daily (WSM Günlük Analiz ekranı)
dfs['daily'].head()"""),

code("""# 1.6 Sayısal kolonların özeti — daily
dfs['daily'].describe().round(2)"""),

md("""---
## BÖLÜM 2 — Tek tablo grafikleri (BASİT)

Henüz tablo birleştirmiyoruz; sadece dağılımları gör."""),

code("""# 2.1 Günlük fark (kayıp/kazanç) dağılımı
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
daily['fark'].hist(bins=60, ax=axes[0], edgecolor='k', alpha=0.75)
axes[0].set_xlabel('Fark (lt)'); axes[0].set_title('Günlük fark dağılımı')
daily['satis'].hist(bins=60, ax=axes[1], edgecolor='k', alpha=0.75, color='steelblue')
axes[1].set_xlabel('Satış (lt)'); axes[1].set_title('Günlük satış dağılımı')
plt.tight_layout(); plt.show()"""),

code("""# 2.2 Alarm oranı — kaç gün SEL aşıldı?
alarm_oran = daily['alarm'].mean()
print(f'Alarm oranı: {alarm_oran:.1%}  ({daily.alarm.sum()} / {len(daily)} gün-tank)')
daily.groupby('alarm').size()"""),

code("""# 2.3 Ürün türüne göre satış
daily.groupby('akaryakit_turu')['satis'].agg(['count','mean','sum']).round(1)"""),

code("""# 2.4 Null kontrolü — hangi kolonlarda eksik var?
null_rapor = []
for name, df in dfs.items():
    for col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            null_rapor.append({'tablo': name, 'kolon': col, 'null': n,
                               'oran_%': round(100*n/len(df), 2)})
null_df = pd.DataFrame(null_rapor).sort_values('null', ascending=False)
print(f'Toplam null içeren kolon: {len(null_df)}')
display(null_df)"""),

md("""---
## BÖLÜM 3 — Bir istasyon + bir tank seç (ORTA)

Gerçek sistemde yaptığın gibi: **istasyon seç → tank seç → detay sekmeleri**.

Aşağıdaki `IST` ve `TANK` değerlerini değiştirerek başka tankları da inceleyebilirsin."""),

code("""# 3.1 Parametreler — burayı değiştir
IST = 'IST_001'   # istasyon kodu
TANK = 1          # tank numarası

tank_info = filter_tank(dfs['tanks'], IST, TANK)
tank_daily  = filter_tank(dfs['daily'], IST, TANK).sort_values('tarih')
tank_ue1t   = filter_tank(dfs['ue1t_30min'], IST, TANK).sort_values('saat_1')
tank_tx     = filter_tank(dfs['transactions'], IST, TANK).sort_values('satis_zamani')
tank_inv    = filter_tank(dfs['inventory_30min'], IST, TANK).sort_values('envanter_tarihi')
tank_deliv  = filter_tank(dfs['deliveries'], IST, TANK)

print(f'Seçilen: {IST} — Tank {TANK}')
print('Tank bilgisi:')
display(tank_info.T)
print(f'\\nSatır sayıları: daily={len(tank_daily)}, ue1t={len(tank_ue1t)}, '
      f'tx={len(tank_tx)}, inv={len(tank_inv)}, deliv={len(tank_deliv)}')"""),

code("""# 3.2 Seçilen tank — 90 günlük özet grafik
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
tank_daily.plot(x='tarih', y='satis', ax=axes[0], legend=False, color='steelblue')
axes[0].set_ylabel('Satış (lt)'); axes[0].set_title(f'{IST} Tank {TANK} — Günlük özet')
tank_daily.plot(x='tarih', y='fark', ax=axes[1], legend=False, color='coral')
axes[1].axhline(0, color='k', lw=0.5); axes[1].set_ylabel('Fark (lt)')
tank_daily.plot(x='tarih', y='kapanis', ax=axes[2], legend=False, color='green')
axes[2].set_ylabel('Kapanış stok (lt)')
tank_daily.plot(x='tarih', y='alarm', ax=axes[3], legend=False, drawstyle='steps-post', color='red')
axes[3].set_ylabel('Alarm'); axes[3].set_xlabel('Tarih')
plt.tight_layout(); plt.show()"""),

code("""# 3.3 En yüksek |fark| gününü bul — o günü derinlemesine incele
idx = tank_daily['fark'].abs().idxmax()
row = tank_daily.loc[idx]
GUN = row['tarih']
print(f'En dikkat çekici gün: {GUN.date()}')
print(f'  Satış={row.satis:.1f}  Fark={row.fark:.1f}  Alarm={row.alarm}  SEL={row.sel:.1f}')

u_gun = tank_ue1t[tank_ue1t.saat_1.dt.normalize() == GUN]
t_gun = tank_tx[tank_tx.satis_zamani.dt.normalize() == GUN]
print(f'  30dk dönem sayısı: {len(u_gun)}  |  tekil satış: {len(t_gun)}')"""),

code("""# 3.4 Seçilen gün — 30 dk kayıp/kazanç grafiği
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
u_gun.plot(x='saat_1', y='donem_sonu_stok', ax=axes[0], legend=False)
axes[0].set_ylabel('Stok (lt)')
u_gun.plot(x='saat_1', y='pompa_satis', ax=axes[1], legend=False, color='orange')
axes[1].set_ylabel('30dk satış (lt)')
u_gun.plot(x='saat_1', y='kayip_kazanc', ax=axes[2], legend=False, color='red')
axes[2].axhline(0, color='k', lw=0.5); axes[2].set_ylabel('Kayıp/Kazanç (lt)')
axes[2].set_xlabel('Saat')
fig.suptitle(f'{IST} T{TANK} — {GUN.date()} detay', y=1.01)
plt.tight_layout(); plt.show()"""),

md("""---
## BÖLÜM 4 — Tablolar arası tutarlılık (ORTA → İLERİ)

Gerçek WSM'de en önemli mühendislik kontrolü: **katmanlar birbirini tutuyor mu?**

```
transactions (tekil)  →  30dk topla  →  ue1t.pompa_satis
ue1t (30dk)           →  gün topla   →  daily.satis
```"""),

code("""# 4.1 Tek gün, tek tank — üç katman karşılaştırması
d_val = tank_daily[tank_daily.tarih == GUN]['satis'].iloc[0]
u_val = u_gun['pompa_satis'].sum()
t_val = t_gun['litre'].sum()

print('=' * 50)
print(f'{IST} Tank {TANK} — {GUN.date()} SATIŞ KARŞILAŞTIRMASI')
print('=' * 50)
print(f'  daily.satis           : {d_val:,.2f} lt')
print(f'  ue1t toplam (30dk)    : {u_val:,.2f} lt')
print(f'  transactions toplam   : {t_val:,.2f} lt')
print(f'  daily vs ue1t fark    : {d_val - u_val:,.4f} lt')
print(f'  ue1t vs tx fark       : {u_val - t_val:,.2f} lt')
if abs(d_val - u_val) < 0.01:
    print('\\n✓ daily ↔ ue1t TUTARLI')
if abs(u_val - t_val) < 0.01:
    print('✓ ue1t ↔ transactions TUTARLI')
else:
    print('⚠ ue1t ↔ transactions farkı → unmapped satış (tank_no null) olabilir')"""),

code("""# 4.2 UE1T mutabakat denklemi — her satırda tutmalı
# donem_sonu = donem_basi + dolum - satis + kayip
res = (tank_ue1t['donem_sonu_stok']
       - (tank_ue1t['donem_basi_stok'] + tank_ue1t['tanka_dolum']
          - tank_ue1t['pompa_satis'] + tank_ue1t['kayip_kazanc']))
print('Mutabakat denklemi max |residual|:', res.abs().max())
print('(0.0000 olmalı)')"""),

code("""# 4.3 Tüm veri seti — otomatik tutarlılık kontrolü
checks = run_all_checks(dfs)
print('Tüm veri seti özet kontroller:')
for k, v in checks['summary'].items():
    print(f'  {k}: {v}')"""),

code("""# 4.4 TX → UE1T fark haritası (unmapped satış avcılığı)
from utils.validation import check_tx_to_ue1t
txu = check_tx_to_ue1t(dfs['transactions'], dfs['ue1t_30min'])
mismatch = txu[txu['tx_ue1t_fark'].abs() > 0.01]
print(f'TX↔UE1T uyuşmayan 30dk dönem: {len(mismatch):,}')
print(f'Unmapped satış (tank_no null): {dfs["transactions"].tank_no.isna().sum():,} işlem')
print(f'Unmapped litre toplamı: {dfs["transactions"].loc[dfs["transactions"].tank_no.isna(),"litre"].sum():,.1f} lt')"""),

md("""---
## BÖLÜM 5 — Tabloları birleştir / JOIN (İLERİ)

Aynı tank, aynı zaman — farklı sekmelerden bilgi bir arada."""),

code("""# 5.1 UE1T + Envanter join (saat_2 = envanter_tarihi)
merged = tank_ue1t.merge(
    tank_inv,
    left_on=['istasyon_kodu', 'tank_no', 'saat_2'],
    right_on=['istasyon_kodu', 'tank_no', 'envanter_tarihi'],
    how='left',
    suffixes=('_ue1t', '_inv')
)
print('Join sonrası:', merged.shape)
merged[['saat_1','donem_sonu_stok','kayip_kazanc','sicaklik_ue1t',
        'su_seviyesi_cm','urun_miktari_brut']].head()"""),

code("""# 5.2 Stok + sıcaklık + su — çift eksen grafik
fig, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(merged['saat_1'], merged['donem_sonu_stok'], color='steelblue', label='Stok')
ax1.set_ylabel('Stok (lt)', color='steelblue')
ax2 = ax1.twinx()
ax2.plot(merged['saat_1'], merged['sicaklik_ue1t'], color='orange', alpha=0.6, label='Sıcaklık')
ax2.set_ylabel('Sıcaklık (°C)', color='orange')
ax1.set_title(f'{IST} T{TANK} — Stok ve sıcaklık (UE1T + Envanter join)')
plt.tight_layout(); plt.show()"""),

code("""# 5.3 Brüt vs Net hacim farkı (sıcaklık kompanzasyonu)
merged2 = merged.dropna(subset=['urun_miktari_brut','urun_miktari_net']).copy()
merged2['brut_net_fark'] = merged2['urun_miktari_brut'] - merged2['urun_miktari_net']
print('Brüt-Net fark istatistikleri:')
print(merged2['brut_net_fark'].describe().round(3))"""),

code("""# 5.4 Mapping join — satışlar hangi pompa/tabancadan?
mapping = dfs['mapping']
ist_mapping = mapping[mapping['istasyon_kodu'] == IST]
print(f'{IST} pompa/tabanca eşlemesi:')
display(ist_mapping.head(10))

# Bu tanka bağlı tabancalar
print(f'\\nTank {TANK} tabancaları:')
display(ist_mapping[ist_mapping['tank_no'] == TANK])"""),

code("""# 5.5 Dolum olayları — deliveries tablosu
if len(tank_deliv):
    print(f'{IST} T{TANK} — {len(tank_deliv)} dolum kaydı')
    display(tank_deliv[['dolum_baslangic','dolum_net','dolum_oncesi_hacim',
                         'dolum_sonrasi_hacim','sicaklik']].head())
else:
    print('Bu tank için dolum kaydı yok.')"""),

md("""---
## BÖLÜM 6 — Null ve veri kalitesi (İLERİ)

Eksik veriler bilinçli — neden eksik olduklarını bul."""),

code("""# 6.1 Seçilen tankta null var mı?
for name, df in [('transactions', tank_tx), ('ue1t', tank_ue1t), ('inventory', tank_inv)]:
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f'{name}:', nulls.to_dict())
    else:
        print(f'{name}: null yok')"""),

code("""# 6.2 Sıcaklık null — hangi günlerde kümeleniyor?
sic_null = tank_ue1t[tank_ue1t['sicaklik'].isna()].copy()
if len(sic_null):
    sic_null['gun'] = sic_null['saat_1'].dt.date
    print('Sıcaklık null günleri (probe arızası adayı):')
    print(sic_null.groupby('gun').size())
else:
    print('Bu tankta sıcaklık null yok.')"""),

code("""# 6.3 Merkeze geliş gecikmesi — router arızası adayı
inv2 = tank_inv.dropna(subset=['merkeze_gelis_tarihi']).copy()
inv2['gecikme_dk'] = (inv2['merkeze_gelis_tarihi'] - inv2['envanter_tarihi']).dt.total_seconds() / 60
print('Gecikme istatistikleri (dk):')
print(inv2['gecikme_dk'].describe().round(1))
inv2.nlargest(5, 'gecikme_dk')[['envanter_tarihi','gecikme_dk']]"""),

code("""# 6.4 Unmapped satış — tank_no null
unmapped = tank_tx[tank_tx['tank_no'].isna()]
print(f'Unmapped işlem: {len(unmapped)} / {len(tank_tx)} ({100*len(unmapped)/max(len(tank_tx),1):.1f}%)')
if len(unmapped):
    print('Toplam unmapped litre:', unmapped['litre'].sum())"""),

md("""---
## BÖLÜM 7 — Anomali sinyalleri keşfi (İLERİ)

Ground truth kullanmadan kural tabanlı sinyaller — ML öncesi feature adayları."""),

code("""# 7.1 Feature adayları türet
u = tank_ue1t.copy()
u['kk_oran'] = np.where(u['pompa_satis'] > 1, u['kayip_kazanc'] / u['pompa_satis'], np.nan)
u['satisiz'] = (u['pompa_satis'] == 0).astype(int)
u['gece'] = u['saat_1'].dt.hour.between(0, 5).astype(int)
u[['kayip_kazanc','kk_oran','satisiz','gece']].describe().round(3)"""),

code("""# 7.2 Sinyal: Satışsız kayıp (statik sızıntı adayı)
satisiz_kayip = u[(u['pompa_satis'] == 0) & (u['kayip_kazanc'] < -5)]
print(f'Satışsız kayıp >5 lt dönem: {len(satisiz_kayip)}')
if len(satisiz_kayip):
    display(satisiz_kayip.nsmallest(5, 'kayip_kazanc')[
        ['saat_1','kayip_kazanc','donem_basi_stok','donem_sonu_stok']])"""),

code("""# 7.3 Sinyal: Gece satışsız düşüş (pompacı manipülasyon adayı)
gece = u[(u['gece'] == 1) & (u['pompa_satis'] == 0) & (u['kayip_kazanc'] < -20)]
print(f'Gece satışsız düşüş >20 lt: {len(gece)}')
if len(gece):
    display(gece[['saat_1','kayip_kazanc']])"""),

code("""# 7.4 Sinyal: Pompa decimal — litre vs tutar/birim_fiyat
tx_ok = tank_tx.dropna(subset=['litre','tutar','birim_fiyat'])
tx_ok = tx_ok[tx_ok['birim_fiyat'] > 0].copy()
tx_ok['litre_hesap'] = tx_ok['tutar'] / tx_ok['birim_fiyat']
tx_ok['litre_oran'] = tx_ok['litre'] / tx_ok['litre_hesap']
decimal = tx_ok[(tx_ok['litre_oran'] > 5) | (tx_ok['litre_oran'] < 0.2)]
print(f'Decimal şüpheli işlem (litre_oran ≠ 1): {len(decimal)}')
if len(decimal):
    print('Oran medyan:', decimal['litre_oran'].median())"""),

code("""# 7.5 Sinyal: Su seviyesi sıçraması (su faktörü adayı)
inv3 = tank_inv.sort_values('envanter_tarihi').copy()
inv3['su_diff'] = inv3['su_seviyesi_cm'].diff()
su_spike = inv3[inv3['su_diff'] > 0.05]
print(f'Su sıçrama (>0.05 cm/30dk): {len(su_spike)}')
if len(su_spike):
    display(su_spike.nlargest(5, 'su_diff')[['envanter_tarihi','su_seviyesi_cm','su_diff']])"""),

code("""# 7.6 Manifold mu? — tanks tablosundan kontrol
if tank_info['is_manifold'].iloc[0] == 1:
    grp = tank_info['manifold_grup_no'].iloc[0]
    partner = dfs['tanks'][(dfs['tanks'].istasyon_kodu==IST) &
                           (dfs['tanks'].manifold_grup_no==grp) &
                           (dfs['tanks'].tank_no!=TANK)]
    if len(partner):
        pno = partner['tank_no'].iloc[0]
        print(f'Manifold çift: Tank {TANK} ↔ Tank {pno}')
        u2 = filter_tank(dfs['ue1t_30min'], IST, int(pno)).sort_values('saat_1')
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(u['saat_1'], u['kayip_kazanc'].cumsum(), label=f'T{TANK}')
        ax.plot(u2['saat_1'], u2['kayip_kazanc'].cumsum(), label=f'T{pno}')
        ax.set_ylabel('Kümülatif kayıp/kazanç'); ax.legend()
        ax.set_title(f'{IST} Manifold çift — kümülatif fark')
        plt.tight_layout(); plt.show()
else:
    print('Bu tank manifoldlu değil.')"""),

md("""---
## BÖLÜM 8 — Özet ve sonraki adımlar

### Bu notebook'ta yaptıkların:
1. ✅ 8 tabloyu yükledin ve tanıdın
2. ✅ Tek tablo dağılımları ve null analizi
3. ✅ Bir istasyon+tank seçip 90 günlük profil çıkardın
4. ✅ Katman tutarlılığını doğruladın (tx → ue1t → daily)
5. ✅ Join ile envanter + UE1T birleştirdin
6. ✅ Anomali sinyalleri keşfettin (ground truth kullanmadan)

### Sonraki adımlar (sen yapacaksın):
| Adım | Ne | Nerede |
|---|---|---|
| Feature engineering | kk_oran, tx_ue1t_fark, su_diff → feature tablosu | Yeni notebook |
| Null doldurma | sicaklik, tank_no, birim_fiyat stratejileri | 03 veya yeni |
| ML | CatBoost / IsolationForest | ground_truth ile karşılaştır |
| Raporlama | Bulguları grafik + tablo olarak staj defterine yaz | Word/PDF |

### Deftere yazılacak cümle:
> "Sentetik wetstock veri ambarında 8 ilişkili tablo üzerinde keşifsel veri analizi yapıldı.
> Katman tutarlılığı (transactions→ue1t→daily) doğrulandı. Null desenleri ve
> kural tabanlı anomali sinyalleri incelendi."

---

**İpucu:** `IST` ve `TANK` değişkenlerini değiştirip notebook'u baştan çalıştır — farklı tankların davranışını karşılaştır."""),

code("""# BONUS — Hızlı karşılaştırma: 3 tank yan yana fark grafiği
fig, ax = plt.subplots(figsize=(14, 4))
for tno in [1, 2, 3]:
    td = filter_tank(dfs['daily'], IST, tno)
    ax.plot(td['tarih'], td['fark'], label=f'Tank {tno}', alpha=0.8)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('Günlük fark (lt)'); ax.set_xlabel('Tarih')
ax.set_title(f'{IST} — Tank 1/2/3 fark karşılaştırması')
ax.legend()
plt.tight_layout(); plt.show()"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Oluşturuldu:", OUT)
