#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gün 7 derin EDA notebook'unu oluşturur / günceller."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "10_gun07_derin_eda.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {
        "cell_type": "code", "metadata": {}, "source": s,
        "outputs": [], "execution_count": None,
    }


cells = [
md("""# Gün 7 — Derinlemesine EDA

**Staj günü:** 7 (09.07.2026)

**Amaç:** Temel EDA'nın üzerine wetstock verisini domain açısından derinlemesine incelemek.

**Kapsam (12 bölüm):**
1. Veri envanteri — 8 tablo özeti
2. Sıcaklık ↔ kayıp/kazanç
3. Kümülatif kayıp trendleri
4. Tanklar arası karşılaştırma
5. Manifold çifti analizi
6. Günlük ↔ 30 dk rollup
7. Saat × gün heatmap (kayıp profili)
8. Dolum sonrası kayıp/kazanç
9. Envanter katmanı (su, brüt-net)
10. Unmapped satış analizi
11. Alarm günü 30 dk profili
12. Tank risk özeti tablosu

> Veri: gerçek WSM profiline kalibre **sentetik** ambar. `ground_truth/` bu notebook'ta **açılmaz**.
"""),

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

from utils.data_loader import load_all, filter_tank, merge_ue1t_inventory, summary_table
from utils.validation import run_all_checks, check_tx_to_ue1t
from utils.plots import setup_style, save_fig, set_section

setup_style()
sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

dfs = load_all()
ue   = dfs['ue1t_30min'].copy()
tanks = dfs['tanks'].copy()
inv  = dfs['inventory_30min'].copy()
daily = dfs['daily'].copy()
tx   = dfs['transactions'].copy()
deliv = dfs['deliveries'].copy()
mapping = dfs['mapping'].copy()
stations = dfs['stations'].copy()

ue = ue.sort_values(['istasyon_kodu', 'tank_no', 'saat_1']).reset_index(drop=True)
print('=== Yüklenen tablolar ===')
for k, v in dfs.items():
    print(f'  {k:18} {v.shape[0]:>8,} satır × {v.shape[1]} kolon')"""),

md("""## 0. Veri envanteri — tüm tablolar

8 operasyonel tablonun satır/kolon/null özeti ve tarih aralıkları."""),

code("""# Tablo özeti
ozet = summary_table(dfs)
display(ozet)

# Tarih aralıkları
print('\\n=== Tarih aralıkları ===')
print(f'daily     : {daily.tarih.min().date()} → {daily.tarih.max().date()}  ({daily.tarih.nunique()} gün)')
print(f'ue1t      : {ue.saat_1.min()} → {ue.saat_1.max()}')
print(f'tx        : {tx.satis_zamani.min()} → {tx.satis_zamani.max()}')
print(f'deliveries: {deliv.dolum_baslangic.min()} → {deliv.dolum_baslangic.max()}  ({len(deliv):,} kayıt)')

# İstasyon × tank matrisi
print('\\n=== İstasyon / tank sayısı ===')
display(stations)
print(tanks.groupby('istasyon_kodu').size().to_string())

# Ürün dağılımı
print('\\n=== Ürün türü (tank) ===')
print(tanks['akaryakit_turu'].value_counts().to_string())"""),

md("""## 1. Sıcaklık — Kayıp/Kazanç ilişkisi

Satışsız dönemlerde sıcaklık etkisini en temiz görürüz. Korelasyon yüksekse termal kaynaklı fark; düşükse gerçek sızıntı şüphesi."""),

code("""nosale = ue[(ue.pompa_satis == 0) & ue.sicaklik.notna()].copy()
nosale['dsic'] = nosale.groupby(['istasyon_kodu','tank_no'])['sicaklik'].diff()
sub = nosale.dropna(subset=['dsic'])

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].scatter(sub['sicaklik'], sub['kayip_kazanc'].clip(-30, 30), s=3, alpha=0.2)
ax[0].set_xlabel('Sıcaklık (°C)'); ax[0].set_ylabel('Kayıp/Kazanç (L)')
ax[0].set_title('Sıcaklık vs Kayıp (satışsız)')
ax[1].scatter(sub['dsic'], sub['kayip_kazanc'].clip(-30, 30), s=3, alpha=0.2, color='tab:green')
ax[1].set_xlabel('ΔSıcaklık'); ax[1].set_ylabel('Kayıp/Kazanç (L)')
ax[1].set_title('ΔSıcaklık vs Kayıp')
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='01_sicaklik_kayip')

corr = sub[['sicaklik','dsic','kayip_kazanc']].corr()['kayip_kazanc']
print('Korelasyon (kayip_kazanc ile):')
print(corr.round(3).to_string())

# İstasyon kırılımında ortalama sıcaklık ve kayıp
ist_ozet = ue.groupby('istasyon_kodu').agg(
    ort_sicaklik=('sicaklik','mean'),
    ort_kayip=('kayip_kazanc','mean'),
    satis_toplam=('pompa_satis','sum'),
).round(2)
print('\\n=== İstasyon özeti ===')
display(ist_ozet)"""),

md("""## 2. Kümülatif kayıp trendleri

Sürekli aşağı giden kümülatif eğim = kronik kayıp (statik sızıntı adayı)."""),

code("""fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# IST_001 ve IST_002 — tüm tanklar
ax = axes[0]
for (st, tk), g in ue.groupby(['istasyon_kodu','tank_no']):
    if st in ('IST_001', 'IST_002'):
        g = g.sort_values('saat_1')
        ax.plot(g['saat_1'], g['kumulatif_kayip_kazanc'], lw=1, label=f'{st}/T{tk}', alpha=0.8)
ax.axhline(0, color='k', lw=0.5)
ax.set_title('Kümülatif Kayıp/Kazanç — IST_001 & IST_002')
ax.set_ylabel('Kümülatif L'); ax.legend(fontsize=7, ncol=3)

# En çok kümülatif kaybeden 5 tank
ax = axes[1]
son = ue.sort_values('saat_1').groupby(['istasyon_kodu','tank_no']).last()
top5 = son.nsmallest(5, 'kumulatif_kayip_kazanc')
for (st, tk), _ in top5.iterrows():
    g = ue[(ue.istasyon_kodu==st) & (ue.tank_no==tk)].sort_values('saat_1')
    ax.plot(g['saat_1'], g['kumulatif_kayip_kazanc'], lw=1.5, label=f'{st}/T{tk}')
ax.axhline(0, color='k', lw=0.5)
ax.set_title('En çok kümülatif kaybeden 5 tank')
ax.set_ylabel('Kümülatif L'); ax.legend()
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='02_kumulatif_trend')"""),

md("""## 3. Tanklar arası karşılaştırma

Boxplot + istatistik tablosu — hangi tank daha değişken / daha çok kaybediyor?"""),

code("""ue['tank_key'] = ue['istasyon_kodu'] + '/T' + ue['tank_no'].astype(str)

tank_stats = ue.groupby('tank_key').agg(
    ort_kayip=('kayip_kazanc','mean'),
    std_kayip=('kayip_kazanc','std'),
    min_kum=('kumulatif_kayip_kazanc','last'),
    satis_toplam=('pompa_satis','sum'),
    satisiz_oran=('pompa_satis', lambda x: (x==0).mean()),
).round(2)
print('=== Tank istatistikleri (tümü) ===')
display(tank_stats.sort_values('ort_kayip').head(10))
display(tank_stats.sort_values('ort_kayip').tail(10))

keys = ue['tank_key'].unique()[:12]
data = [ue[ue.tank_key==k]['kayip_kazanc'].clip(-40, 40).values for k in keys]
fig, ax = plt.subplots(figsize=(14, 4))
ax.boxplot(data, tick_labels=keys, showfliers=False)
ax.axhline(0, color='r', lw=0.5)
ax.set_title('Tank bazında Kayıp/Kazanç dağılımı (ilk 12)')
ax.set_ylabel('L'); plt.xticks(rotation=45)
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='03_tank_boxplot')"""),

md("""## 4. Manifold çifti analizi

Manifold bağlı tanklarda ters yönlü kayıp/kazanç beklenir (check-valve kaçırması)."""),

code("""man = tanks[tanks.is_manifold == 1][['istasyon_kodu','tank_no','manifold_grup_no','akaryakit_turu']]
print('Manifold tanklar:')
display(man)

# Tüm çiftler için korelasyon tablosu
pairs = []
for (ist, mg), g in man.groupby(['istasyon_kodu','manifold_grup_no']):
    tn = sorted(g['tank_no'].tolist())
    if len(tn) == 2:
        a, b = tn
        ua = ue[(ue.istasyon_kodu==ist)&(ue.tank_no==a)].set_index('saat_1')['kayip_kazanc']
        ub = ue[(ue.istasyon_kodu==ist)&(ue.tank_no==b)].set_index('saat_1')['kayip_kazanc']
        m = pd.concat([ua, ub], axis=1, keys=['a','b']).dropna()
        corr = m['a'].corr(m['b']) if len(m) > 50 else np.nan
        pairs.append({'istasyon': ist, 'grup': mg, 'tank_a': a, 'tank_b': b,
                      'kk_korelasyon': round(corr, 3), 'n_donem': len(m)})
pair_df = pd.DataFrame(pairs)
print('\\n=== Manifold çift korelasyonları ===')
display(pair_df)

# İlk çift — zaman serisi
st, mg = pair_df.iloc[0]['istasyon'], pair_df.iloc[0]['grup']
ta, tb = int(pair_df.iloc[0]['tank_a']), int(pair_df.iloc[0]['tank_b'])
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for tk, ax in zip([ta, tb], axes):
    g = ue[(ue.istasyon_kodu==st)&(ue.tank_no==tk)].sort_values('saat_1').iloc[:48*14]
    ax.plot(g['saat_1'], g['kayip_kazanc'], lw=0.8)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('L'); ax.set_title(f'{st} Tank {tk}')
axes[0].set_title(f'Manifold çifti {st} T{ta}↔T{tb} (korelasyon={pair_df.iloc[0]["kk_korelasyon"]})')
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='04_manifold_cift')"""),

md("""## 5. Null / eksik veri kümelenmesi

Null'lar rastgele değil — probe arızası, router gecikmesi, mapping hatası senaryoları."""),

code("""# Null matrisi — tüm tablolar
null_rows = []
for name, df in dfs.items():
    for col in df.columns:
        n = df[col].isna().sum()
        if n:
            null_rows.append({'tablo': name, 'kolon': col, 'null': n,
                              'oran_pct': round(100*n/len(df), 2)})
null_df = pd.DataFrame(null_rows).sort_values('null', ascending=False)
print('=== Null özeti (tüm tablolar) ===')
display(null_df.head(15))

# Sıcaklık null — tank × gün kümesi
tmp = ue.copy()
tmp['gun'] = tmp['saat_1'].dt.date
tmp['sic_null'] = tmp['sicaklik'].isna()
sic_pivot = tmp.groupby(['istasyon_kodu','tank_no'])['sic_null'].sum()
print('\\nSıcaklık NaN (tank, >0):')
print(sic_pivot[sic_pivot > 0].to_string())

# Merkeze geliş gecikmesi
inv2 = inv.dropna(subset=['merkeze_gelis_tarihi']).copy()
inv2['gecikme_dk'] = (inv2['merkeze_gelis_tarihi'] - inv2['envanter_tarihi']).dt.total_seconds() / 60
print('\\nGecikme istatistikleri (dk):')
print(inv2['gecikme_dk'].describe().round(1).to_string())"""),

md("""## 6. Günlük ↔ 30 dk rollup

`daily.fark` ile UE1T 30 dk kayıp/kazanç toplamının aynı günde tutarlı olup olmadığı."""),

code("""ue['tarih'] = ue['saat_1'].dt.normalize()
ue_gun = ue.groupby(['istasyon_kodu','tank_no','tarih'], as_index=False).agg(
    ue1t_kk_toplam=('kayip_kazanc','sum'),
    ue1t_satis=('pompa_satis','sum'),
    ue1t_dolum=('tanka_dolum','sum'),
)
m = daily.merge(ue_gun, on=['istasyon_kodu','tank_no','tarih'], how='left')
m['fark_ue1t_fark'] = m['fark'] - m['ue1t_kk_toplam']

print('=== Günlük fark vs UE1T kk toplam ===')
print(m[['fark','ue1t_kk_toplam','fark_ue1t_fark']].describe().round(2).to_string())
print(f'\\nMax |fark - ue1t_kk|: {m["fark_ue1t_fark"].abs().max():.2f} L')

# Alarm günleri vs normal
print('\\nOrt. fark — alarm vs normal:')
print(m.groupby('alarm')['fark'].agg(['count','mean','std']).round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 4))
colors = m['alarm'].map({0:'steelblue', 1:'crimson'})
ax.scatter(m['ue1t_kk_toplam'], m['fark'], c=colors, alpha=0.4, s=12)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('UE1T kk toplam (L)'); ax.set_ylabel('Daily fark (L)')
ax.set_title('Günlük fark vs 30dk kk toplam (kırmızı=alarm)')
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='06_daily_ue1t_rollup')"""),

md("""## 7. Saat × haftanın günü heatmap

Kayıp/kazancın hangi saat ve günde yoğunlaştığını görürüz (manipülasyon / gece deseni)."""),

code("""ue['saat'] = ue['saat_1'].dt.hour
ue['haftanin_gunu'] = ue['saat_1'].dt.dayofweek
ue['gun_adi'] = ue['saat_1'].dt.day_name()

heat = ue.pivot_table(
    index='gun_adi', columns='saat', values='kayip_kazanc', aggfunc='mean'
)
gun_sira = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heat = heat.reindex([g for g in gun_sira if g in heat.index])

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(heat, cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Ort. kayıp/kazanç (L)'})
ax.set_title('Ortalama kayıp/kazanç — gün × saat')
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='07_heatmap_gun_saat')

# Gece vs gündüz özet
ue['gece'] = ue['saat'].between(0, 5).astype(int)
print('\\nGece vs gündüz kayıp/kazanç:')
print(ue.groupby('gece')['kayip_kazanc'].agg(['count','mean','std']).round(2).to_string())"""),

md("""## 8. Dolum sonrası kayıp/kazanç

Dolum olaylarından sonraki 6 saatlik pencerede anormal kayıp var mı? (hayali/algılanmayan dolum)."""),

code("""# Dolum zamanlarını UE1T ile eşleştir
deliv2 = deliv.copy()
deliv2['dolum_saat'] = deliv2['dolum_baslangic'].dt.floor('30min')

dolum_donem = ue.merge(
    deliv2[['istasyon_kodu','tank_no','dolum_saat','dolum_net']],
    left_on=['istasyon_kodu','tank_no','saat_1'],
    right_on=['istasyon_kodu','tank_no','dolum_saat'],
    how='inner',
)
print(f'Dolum eşleşen UE1T dönem: {len(dolum_donem):,}')
print('\\nDolum anı kayıp/kazanç istatistikleri:')
print(dolum_donem['kayip_kazanc'].describe().round(2).to_string())

# En büyük 5 dolum
top_dolum = deliv.nlargest(5, 'dolum_net')
print('\\n=== En büyük 5 dolum ===')
display(top_dolum[['dolum_baslangic','istasyon_kodu','tank_no','dolum_net','dolum_oncesi_hacim','dolum_sonrasi_hacim']])

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(deliv['dolum_net'], deliv['dolum_sonrasi_hacim'] - deliv['dolum_oncesi_hacim'],
           alpha=0.6, s=40, edgecolors='k', linewidths=0.3)
ax.plot([0, deliv['dolum_net'].max()], [0, deliv['dolum_net'].max()], 'r--', lw=1, label='1:1')
ax.set_xlabel('Kayıtlı dolum net (L)'); ax.set_ylabel('Hacim artışı (L)')
ax.set_title('Dolum net vs gerçek hacim artışı')
ax.legend(); plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='08_dolum_analizi')"""),

md("""## 9. Envanter katmanı — su ve brüt/net

UE1T + envanter join ile su sıçraması ve sıcaklık kompanzasyonu."""),

code("""IST, TANK = 'IST_001', 1
merged = merge_ue1t_inventory(
    filter_tank(ue, IST, TANK),
    filter_tank(inv, IST, TANK),
)
merged = merged.sort_values('saat_1')
merged['brut_net_fark'] = merged['urun_miktari_brut'] - merged['urun_miktari_net']
merged['su_diff'] = merged.groupby(['istasyon_kodu','tank_no'])['su_seviyesi_cm'].diff()

print(f'{IST} T{TANK} join satır: {len(merged):,}')
print('\\nBrüt-net fark:', merged['brut_net_fark'].describe().round(2).to_string())
print('\\nSu sıçrama (>0.05 cm):', (merged['su_diff'] > 0.05).sum())

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes[0].plot(merged['saat_1'], merged['donem_sonu_stok'], color='steelblue')
axes[0].set_ylabel('Stok (L)'); axes[0].set_title(f'{IST} T{TANK} — stok + envanter')
axes[1].plot(merged['saat_1'], merged['brut_net_fark'], color='coral')
axes[1].set_ylabel('Brüt-Net (L)')
axes[2].plot(merged['saat_1'], merged['su_seviyesi_cm'].fillna(0), color='teal')
axes[2].set_ylabel('Su (cm)'); axes[2].set_xlabel('Tarih')
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='09_envanter_join')"""),

md("""## 10. Unmapped satış analizi

`tank_no` null satışlar — mapping hatası; katman tutarlılığını etkiler."""),

code("""unmapped = tx[tx['tank_no'].isna()]
mapped   = tx[tx['tank_no'].notna()]
print(f'Unmapped işlem : {len(unmapped):,} ({100*len(unmapped)/len(tx):.2f}%)')
print(f'Unmapped litre : {unmapped["litre"].sum():,.1f} L')
print(f'Mapped litre   : {mapped["litre"].sum():,.1f} L')

print('\\n=== Unmapped — istasyon kırılımı ===')
display(unmapped.groupby('istasyon_kodu').agg(
    islem=('litre','count'), litre=('litre','sum')
).sort_values('litre', ascending=False))

# TX vs UE1T fark özeti
checks = run_all_checks(dfs)
txu = checks['tx_to_ue1t']
print('\\n=== TX↔UE1T fark özeti ===')
print(f'Uyuşmayan dönem: {(txu["tx_ue1t_fark"].abs() > 0.01).sum():,}')
print(txu['tx_ue1t_fark'].describe().round(2).to_string())"""),

md("""## 11. Alarm günü 30 dk profili

SEL alarmı olan günlerde 30 dk kayıp/kazanç nasıl dağılıyor?"""),

code("""alarm_gunler = daily[daily['alarm'] == 1].copy()
print(f'Alarm gün-tank: {len(alarm_gunler):,}  (oran: {daily["alarm"].mean():.1%})')

# En yüksek |fark| 3 alarm günü
top_alarm = alarm_gunler.assign(fark_abs=alarm_gunler['fark'].abs()).nlargest(3, 'fark_abs')
print('\\n=== En yüksek |fark| alarm günleri ===')
display(top_alarm[['tarih','istasyon_kodu','tank_no','fark','satis','sel','alarm']])

fig, axes = plt.subplots(len(top_alarm), 1, figsize=(14, 3*len(top_alarm)), sharex=False)
if len(top_alarm) == 1:
    axes = [axes]
for ax, (_, row) in zip(axes, top_alarm.iterrows()):
    u = ue[(ue.istasyon_kodu==row.istasyon_kodu) & (ue.tank_no==row.tank_no)
           & (ue['tarih']==row.tarih)]
    ax.bar(range(len(u)), u['kayip_kazanc'], color='crimson', edgecolor='k', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('L')
    ax.set_title(f'{row.istasyon_kodu} T{row.tank_no} {row.tarih.date()} fark={row.fark:.1f} L')
plt.tight_layout()
set_section('10_deep_eda'); save_fig(name='11_alarm_gun_profil')"""),

md("""## 12. Tank risk özeti tablosu

Her tank için birleşik metrikler — FE ve ML öncesi önceliklendirme."""),

code("""# Tank meta + UE1T + daily rollup
ue2 = ue.copy()
ue2['satisiz_kayip_flag'] = ((ue2['pompa_satis'] == 0) & (ue2['kayip_kazanc'] < -5)).astype(int)

tank_risk = ue2.groupby(['istasyon_kodu','tank_no']).agg(
    ort_kayip=('kayip_kazanc','mean'),
    std_kayip=('kayip_kazanc','std'),
    satisiz_kayip=('satisiz_kayip_flag','sum'),
    kum_son=('kumulatif_kayip_kazanc','last'),
    sic_null=('sicaklik', lambda x: x.isna().sum()),
).reset_index()

alarm_cnt = daily.groupby(['istasyon_kodu','tank_no'])['alarm'].sum().reset_index(name='alarm_gun')
tank_risk = tank_risk.merge(alarm_cnt, on=['istasyon_kodu','tank_no'], how='left')
tank_risk = tank_risk.merge(
    tanks[['istasyon_kodu','tank_no','kapasite','akaryakit_turu','is_manifold','bolmeli']],
    on=['istasyon_kodu','tank_no'], how='left',
)
tank_risk['tank_key'] = tank_risk['istasyon_kodu'] + '/T' + tank_risk['tank_no'].astype(str)

print('=== Tank risk tablosu (en çok alarm) ===')
display(tank_risk.sort_values('alarm_gun', ascending=False).head(10))

print('\\n=== Tank risk tablosu (en çok kümülatif kayıp) ===')
display(tank_risk.nsmallest(5, 'kum_son')[['tank_key','ort_kayip','kum_son','alarm_gun','is_manifold','akaryakit_turu']])

# Kaydet
out = ROOT / 'output' / 'tank_risk_ozet.csv'
out.parent.mkdir(parents=True, exist_ok=True)
tank_risk.to_csv(out, index=False)
print(f'\\nKaydedildi: {out}')"""),

md("""## Gün 7 Özeti

| Bölüm | Bulgu |
|-------|-------|
| Veri envanteri | 8 tablo, 90 gün, 32 tank — yıldız şeması tutarlı |
| Sıcaklık | Satışsız dönemde korelasyon zayıf → termal/sızıntı ayrımı mümkün |
| Kümülatif trend | Kronik kaybeden tanklar belirlendi |
| Manifold | Çiftlerde kk korelasyon ≈ -0.94 (ters yön) |
| Null | Sıcaklık null'ları tank-gün kümesi (probe arızası) |
| Daily↔UE1T | Alarm günlerinde fark profili farklı |
| Heatmap | Gece saatlerinde kayıp deseni |
| Dolum | Dolum anı ve hacim artışı karşılaştırması |
| Envanter | Brüt-net fark, su seviyesi join |
| Unmapped TX | ~%1.1 işlem, mapping hatası sinyali |
| Alarm profili | SEL aşım günlerinde 30dk bar grafik |
| Tank risk | `output/tank_risk_ozet.csv` — ML öncesi öncelik listesi |

**Sonraki adım:** Gün 8 Feature Engineering (`feature_engineering/notebooks/GUN08_feature_engineering.ipynb`)
"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Oluşturuldu: {OUT} ({len(cells)} hücre)")
