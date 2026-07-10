# -*- coding: utf-8 -*-
"""
Sentetik Wetstock veri ambarı üreticisi — v3
WSM eğitim sunumlarındaki alarm kategorilerine kalibre, tamamen anonim.

Tablolar (data/):
  Boyut:  stations, tanks (bölmeli+manifold), mapping
  Olgu:   transactions (fiyat/tutar dahil), deliveries,
          inventory_30min (su + brüt/net + merkeze geliş), ue1t_30min, daily

Etiketli anomali kategorileri ve veri imzaları (sunumlardan):
  statik_sizinti        satışsız dönemlerde sabit kayıp
  dinamik_sizinti       satışla orantılı kayıp
  pompa_kalibrasyon     uzun süreli küçük yüzdesel sistematik fark (± yönlü)
  pompaci_manipulasyonu gece saatlerinde satışsız envanter düşüşü
  pompa_decimal         litre 10x hatalı, tutar/birim_fiyat doğru
  manifold_yakit_gecisi check-valve kaçırması: bir tankta kayıp = diğerinde kazanç
  yakit_gecisi          bölmeli tank: dolumda sac baskılanması, sonra geri dönüş
  mapping_hatasi        satış A tankına yazılır, yakıt B tankından çıkar
  samandira_takilmasi   seviye donar (satışlar kazanç verir), çözülünce ani kayıp
  su_faktoru            su seviyesi hızla artar + küçük görünür kazanç
  sicaklik_degisimi     dolum sonrası ani sıcaklık sıçraması → hacimsel fark
  hayali_dolum          dolum kaydı var, envanter artışı yok → dolum kadar kayıp
  algilanmayan_dolum    envanter artışı var, dolum kaydı yok → kazanç
  gun_sonu_dolum        gece yarısına sarkan dolum: 1. gün kazanç, 2. gün kayıp
  test_satisi           satış yansır, yakıt tanka geri dökülür → satış kadar kazanç
  yakit_cekimi          bayi beyanlı tekil çekim: satışsız büyük düşüş
  atg_arizasi           istasyonun TÜM tanklarında veriler 0 gelir
  probe_arizasi         TEK tankta veriler 0/tutarsız gelir

Etiketsiz (EDA ile keşfedilecek) veri kalitesi senaryoları:
  elektrik_arizasi      istasyon genelinde 30dk satırları hiç oluşmaz (satış da durur)
  tankomat_arizasi      00:00 satırı oluşmaz; dönemin satışı sonraki satıra fark yansır
  router_arizasi        veriler oluşur ama merkeze_gelis_tarihi saatlerce gecikir
  probe sıcaklık nulli  sıcaklık NaN (net hacim hesaplanamaz), gün bazında kümeli
  su probu nulli        su kolonları NaN
  unmapped satış        transactions.tank_no NaN (~%1) → tx→ue1t farkı
  dolum detay nulları   deliveries.sicaklik / merkeze_gelis_tarihi NaN

İnvariantlar:
  - ue1t: donem_sonu = donem_basi + dolum - pompa_satis + kayip_kazanc (her satır)
  - normal dönemlerde kayıp yalnız ölçüm gürültüsü + sıcaklık genleşmesi
  - tx→30dk, 30dk→daily, deliveries→daily birebir (yukarıdaki senaryolar hariç —
    sapmaların KENDİSİ etiketli/keşfedilebilir bulgu)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

rng = np.random.default_rng(7)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

START = pd.Timestamp("2026-01-01")
N_DAYS = 90
PPD = 48
N_PERIODS = N_DAYS * PPD

MOTORIN = ("Motorin(Excellium)", 7)
BENZIN = ("KB 95 Oktan Excellium", 16)
BETA = {7: 0.00084, 16: 0.00120}          # hacimsel genleşme 1/°C (15°C ref)
BASE_PRICE = {7: 43.75, 16: 45.20}

# ------------------------------------------------------------------ boyutlar
STATIONS = [("IST_001", "Marmara"), ("IST_002", "Ege"), ("IST_003", "Marmara"),
            ("IST_004", "Ic Anadolu"), ("IST_005", "Ic Anadolu"),
            ("IST_006", "Akdeniz"), ("IST_007", "Karadeniz"), ("IST_008", "Dogu Anadolu")]

# (ist, tank_no, ürün, kapasite, manifold_grp, bolme_grp)
TANK_DEFS = [
    ("IST_001", 1, MOTORIN, 23000, 1, 0), ("IST_001", 2, BENZIN, 7000, 0, 0),
    ("IST_001", 3, MOTORIN, 10000, 1, 0), ("IST_001", 4, BENZIN, 20000, 0, 0),
    ("IST_001", 5, MOTORIN, 30000, 0, 0),
    ("IST_002", 1, MOTORIN, 23000, 0, 0), ("IST_002", 2, BENZIN, 10000, 0, 0),
    ("IST_002", 3, MOTORIN, 20000, 0, 0), ("IST_002", 4, BENZIN, 7000, 0, 0),
    ("IST_002", 5, MOTORIN, 30000, 0, 0),
    ("IST_003", 1, MOTORIN, 23000, 0, 0),
    ("IST_003", 2, BENZIN, 20000, 0, 1), ("IST_003", 3, MOTORIN, 10000, 0, 1),
    ("IST_004", 1, MOTORIN, 30000, 0, 0), ("IST_004", 2, BENZIN, 10000, 1, 0),
    ("IST_004", 3, MOTORIN, 23000, 0, 0), ("IST_004", 4, BENZIN, 10000, 1, 0),
    ("IST_004", 5, MOTORIN, 20000, 0, 0),
    ("IST_005", 1, MOTORIN, 23000, 1, 0), ("IST_005", 2, BENZIN, 7000, 0, 0),
    ("IST_005", 3, MOTORIN, 23000, 1, 0),
    ("IST_006", 1, MOTORIN, 23000, 0, 0), ("IST_006", 2, BENZIN, 20000, 0, 0),
    ("IST_006", 3, MOTORIN, 10000, 0, 0), ("IST_006", 4, BENZIN, 7000, 0, 0),
    ("IST_007", 1, MOTORIN, 30000, 0, 0), ("IST_007", 2, BENZIN, 10000, 1, 0),
    ("IST_007", 3, MOTORIN, 23000, 0, 0), ("IST_007", 4, BENZIN, 10000, 1, 0),
    ("IST_008", 1, MOTORIN, 23000, 0, 0), ("IST_008", 2, BENZIN, 7000, 0, 0),
    ("IST_008", 3, MOTORIN, 20000, 0, 0),
]

tanks_rows = []
for ist, tno, (urun_ad, urun_no), kap, mgrp, bgrp in TANK_DEFS:
    capi = 250 if kap <= 23000 else 285
    uzunluk = round(kap / (np.pi * (capi / 200) ** 2) * 100)
    tanks_rows.append(dict(
        istasyon_kodu=ist, tank_no=tno, urun_no=urun_no, akaryakit_turu=urun_ad,
        kapasite=kap, tank_capi=capi, tank_uzunlugu=uzunluk,
        bolmeli=int(bgrp > 0), bolme_grup_no=bgrp,
        is_manifold=int(mgrp > 0), manifold_grup_no=mgrp))
tanks = pd.DataFrame(tanks_rows)
stations = pd.DataFrame([dict(istasyon_kodu=k, bolge=b,
                              tank_sayisi=int((tanks.istasyon_kodu == k).sum()))
                         for k, b in STATIONS])

mapping_rows = []
for ist, _ in STATIONS:
    pompa_no = 5
    for _, t in tanks[tanks.istasyon_kodu == ist].iterrows():
        for tab in range(1, int(rng.integers(2, 4)) + 1):
            mapping_rows.append(dict(
                istasyon_kodu=ist, pompa_no=pompa_no, tabanca_no=tab,
                tank_no=t.tank_no, urun_no=t.urun_no,
                akaryakit_turu=t.akaryakit_turu, manifold_grup_no=t.manifold_grup_no))
        pompa_no += 1
mapping = pd.DataFrame(mapping_rows)

# günlük fiyat serisi (küçük drift)
price = {u: BASE_PRICE[u] * np.cumprod(1 + rng.normal(0.0004, 0.002, N_DAYS))
         for u in (7, 16)}

# ------------------------------------------------------------------ senaryo planı
tank_keys = [(r.istasyon_kodu, r.tank_no) for r in tanks.itertuples()]
used = set()

def take(n, station=None):
    pool = [k for k in tank_keys if k not in used
            and (station is None or k[0] == station)]
    ks = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
    used.update(ks)
    return ks

def win(lo_start, len_lo, len_hi):
    d0 = int(rng.integers(lo_start, N_DAYS - len_hi - 2))
    return d0, d0 + int(rng.integers(len_lo, len_hi))

# pencere anomalileri: (ist,tno) -> (kategori, d0, d1, param)
window = {}
for k in take(2):
    d0, d1 = win(4, 5, 9); window[k] = ("statik_sizinti", d0, d1, rng.uniform(3, 8))
for k in take(2):
    d0, d1 = win(4, 5, 9); window[k] = ("dinamik_sizinti", d0, d1, rng.uniform(0.015, 0.03))
for k in take(2):
    d0, d1 = win(4, 6, 10)
    window[k] = ("pompa_kalibrasyon", d0, d1,
                 float(rng.choice([-1, 1])) * rng.uniform(0.008, 0.015))
su_tank = take(1)[0]
su_d0, su_d1 = win(6, 3, 5)
stuck_tank = take(1)[0]
stuck_d0, stuck_d1 = win(6, 1, 3)          # şamandıra takılması
decimal_tank = take(1)[0]
decimal_day = int(rng.integers(10, N_DAYS - 5))

# bölmeli çift (IST_003: 2 ve 3) — kullanılmış say
used.update({("IST_003", 2), ("IST_003", 3)})
BOLME = {"IST_003": (2, 3)}

# mapping hatası: aynı istasyonda A<B iki boş tank
map_cfg = None
for ist, _ in sorted(STATIONS, key=lambda _: rng.random()):
    cands = sorted(t for i2, t in tank_keys if i2 == ist and (i2, t) not in used)
    if len(cands) >= 2:
        A, B = cands[0], cands[1]
        used.update({(ist, A), (ist, B)})
        d0, d1 = win(8, 4, 7)
        map_cfg = (ist, A, B, d0, d1)
        break

# pompacı manipülasyonu: 2 tank, pencere içinde gece çekimleri (nokta olayı üretilir)
pompaci_tanks = take(2)
# probe seviye arızası (tek tank, 0 gelen veriler)
probe_tank = take(1)[0]
probe_p0 = int(rng.integers(10 * PPD, N_PERIODS - PPD))
probe_len = int(rng.integers(4, 9))

# istasyon olayları (farklı istasyonlar)
ist_codes = [k for k, _ in STATIONS]
ev_ist = list(rng.choice(ist_codes, size=4, replace=False))
ELEK_IST, ATG_IST, ROUTER_IST, TKM_IST = ev_ist
elek_p0 = int(rng.integers(5 * PPD, N_PERIODS - PPD)); elek_len = int(rng.integers(6, 13))
atg_p0 = int(rng.integers(5 * PPD, N_PERIODS - PPD)); atg_len = int(rng.integers(8, 13))
router_day = int(rng.integers(5, N_DAYS - 2))
tankomat_days = {(TKM_IST, int(rng.integers(3, N_DAYS - 1))) for _ in range(2)}

# nokta olayları: (ist,tno,p) -> list[(tip, param)]
point = defaultdict(list)

def rand_p(day_lo=3, slot_lo=16, slot_hi=44):
    return int(rng.integers(day_lo, N_DAYS - 2)) * PPD + int(rng.integers(slot_lo, slot_hi))

for _ in range(2):
    k = tank_keys[int(rng.integers(len(tank_keys)))]
    point[(k[0], k[1], rand_p())].append(("yakit_cekimi", rng.uniform(300, 900)))
for _ in range(3):
    k = tank_keys[int(rng.integers(len(tank_keys)))]
    point[(k[0], k[1], rand_p())].append(("test_satisi", rng.uniform(30, 70)))
for _ in range(2):
    k = tank_keys[int(rng.integers(len(tank_keys)))]
    point[(k[0], k[1], rand_p(slot_lo=30, slot_hi=44))].append(
        ("hayali_dolum", rng.uniform(3000, 8000)))
for _ in range(2):
    k = tank_keys[int(rng.integers(len(tank_keys)))]
    point[(k[0], k[1], rand_p())].append(("algilanmayan_dolum", rng.uniform(4000, 9000)))
gsd_days = []
for _ in range(2):
    k = tank_keys[int(rng.integers(len(tank_keys)))]
    d = int(rng.integers(3, N_DAYS - 2))
    X = rng.uniform(4000, 8000)
    point[(k[0], k[1], d * PPD + 47)].append(("gsd_fiziksel", X))
    point[(k[0], k[1], (d + 1) * PPD)].append(("gsd_kayit", X))
    gsd_days.append((k, d))
# pompacı gece çekimleri
for k in pompaci_tanks:
    d0, d1 = win(4, 5, 9)
    for d in range(d0, d1):
        if rng.random() < 0.7:
            p = d * PPD + int(rng.integers(3, 9))      # 01:30 - 04:30
            point[(k[0], k[1], p)].append(("pompaci", rng.uniform(40, 150)))

# manifold transferleri
manifold_pairs = []
for (ist, grp), g in tanks[tanks.is_manifold == 1].groupby(["istasyon_kodu", "manifold_grup_no"]):
    tn = sorted(g.tank_no)
    if len(tn) == 2:
        manifold_pairs.append((ist, tn[0], tn[1]))
transfer = {}
for ist, a, b in manifold_pairs:
    for _ in range(int(rng.integers(6, 12))):
        transfer[(ist, int(rng.integers(0, N_PERIODS)))] = \
            (a, b, float(rng.uniform(80, 350)) * rng.choice([-1, 1]))

# null senaryoları
temp_fail = {(tank_keys[int(rng.integers(len(tank_keys)))], int(rng.integers(2, N_DAYS - 1)))
             for _ in range(5)}
water_fail = {(tank_keys[int(rng.integers(len(tank_keys)))], int(rng.integers(2, N_DAYS - 1)))}

# ------------------------------------------------------------------ yardımcılar
def diurnal():
    h = np.arange(PPD) / 2.0
    w = 0.25 + np.exp(-((h - 8.5) ** 2) / 8) + 1.15 * np.exp(-((h - 17.5) ** 2) / 10)
    w[(h >= 1) & (h < 5)] *= 0.3
    return w / w.sum()

W = diurnal()
nozzles = {(r.istasyon_kodu, r.tank_no):
           mapping[(mapping.istasyon_kodu == r.istasyon_kodu) &
                   (mapping.tank_no == r.tank_no)][["pompa_no", "tabanca_no"]].values.tolist()
           for r in tanks.itertuples()}

tx_rows, ue1t_rows, inv_rows, del_rows = [], [], [], []
delivery_id = 0

# ------------------------------------------------------------------ simülasyon
for ist, _ in STATIONS:
    sub = tanks[tanks.istasyon_kodu == ist].sort_values("tank_no")
    S = {}
    for t in sub.itertuples():
        S[t.tank_no] = dict(
            stock=t.kapasite * rng.uniform(0.45, 0.75), temp=20.0,
            su=rng.uniform(0.3, 1.5), cum=0.0, pend=None, rep_prev=None,
            frozen=None, was_special=False,
            target=t.kapasite * rng.uniform(0.13, 0.20),
            lpc=t.kapasite / (t.tank_capi * 0.78), dead=t.kapasite * 0.04,
            reorder=t.kapasite * 0.28, kap=t.kapasite, urun=t.urun_no,
            urun_ad=t.akaryakit_turu, mean_tx=42 if t.urun_no == 7 else 27)

    dyn = defaultdict(list)      # (tno, p) -> [(delta, kategori)]

    for p in range(N_PERIODS):
        day, slot = divmod(p, PPD)
        t0 = START + pd.Timedelta(minutes=30 * p)
        t1 = t0 + pd.Timedelta(minutes=30)
        h = slot / 2.0
        ambient = (20 + 3 * np.sin(2 * np.pi * (h - 9) / 24)
                   + 2.5 * np.sin(2 * np.pi * day / N_DAYS) + rng.normal(0, 0.3))
        elek = (ist == ELEK_IST) and (elek_p0 <= p < elek_p0 + elek_len)
        atg = (ist == ATG_IST) and (atg_p0 <= p < atg_p0 + atg_len)
        tkm_drop = (ist, day) in tankomat_days and slot == 0
        misroute = {}

        for tno in sub.tank_no:
            st = S[tno]
            key = (ist, tno)
            beta = BETA[st["urun"]]
            cat, label = "normal", 0
            wa = window.get(key)
            w_active = wa if (wa and wa[1] <= day < wa[2]) else None
            stuck = (key == stuck_tank) and (stuck_d0 <= day < stuck_d1)
            probe0 = (key == probe_tank) and (probe_p0 <= p < probe_p0 + probe_len)

            if st["rep_prev"] is None:
                st["rep_prev"] = round(st["stock"], 2)

            # --- sıcaklık ve hacimsel genleşme
            new_temp = st["temp"] + 0.06 * (ambient - st["temp"]) + rng.normal(0, 0.05)

            # --- dolum
            dolum = 0.0
            if st["pend"] and st["pend"][0] == p:
                if elek or atg or stuck:
                    st["pend"] = (p + 1, st["pend"][1])
                else:
                    amount = st["pend"][1]; st["pend"] = None
                    dolum = amount
                    st["stock"] += amount
                    shock = rng.uniform(1.5, 4.0) * rng.choice([-1, 1], p=[0.25, 0.75])
                    new_temp += shock
                    delivery_id += 1
                    d_temp = new_temp + rng.uniform(1, 4)
                    del_rows.append(dict(
                        dolum_id=delivery_id,
                        dolum_baslangic=t0 + pd.Timedelta(minutes=int(rng.integers(0, 12))),
                        dolum_bitis=t0 + pd.Timedelta(minutes=int(rng.integers(15, 29))),
                        istasyon_kodu=ist, tank_no=tno, akaryakit_turu=st["urun_ad"],
                        dolum_oncesi_hacim=round(st["stock"] - amount, 2),
                        dolum_sonrasi_hacim=round(st["stock"], 2),
                        dolum_brut=round(amount / (1 - BETA[st["urun"]] * (d_temp - 15)), 2),
                        dolum_net=round(amount, 2),
                        sicaklik=round(d_temp, 2) if rng.random() > 0.10 else np.nan,
                        merkeze_gelis_tarihi=(t1 + pd.Timedelta(hours=int(rng.integers(1, 8)))
                                              if rng.random() > 0.15 else pd.NaT)))
                    # bölmeli tank: sac baskılanması → partnerde geçici kazanç
                    if ist in BOLME and tno in BOLME[ist]:
                        pa, pb = BOLME[ist]
                        partner = pb if tno == pa else pa
                        g = rng.uniform(15, 45)
                        dyn[(partner, p)].append((g, "yakit_gecisi"))
                        for j in range(1, 7):
                            dyn[(partner, p + j)].append((-g / 6, "yakit_gecisi"))
            elif st["pend"] is None and st["stock"] < st["reorder"] and 16 <= slot < 36:
                amount = round(min(st["kap"] * 0.88 - st["stock"],
                                   rng.choice([5000, 8000, 10000, 13000])
                                   * rng.uniform(0.92, 1.0)), 2)
                if amount > 1000:
                    st["pend"] = (p + int(rng.integers(4, 20)), amount)

            dT = new_temp - st["temp"]
            temp_eff = beta * st["stock"] * dT
            st["stock"] += temp_eff
            st["temp"] = new_temp

            # --- satışlar
            litres = np.array([])
            if not elek:
                lam = st["target"] / st["mean_tx"] * W[slot]
                n_tx = rng.poisson(lam)
                if n_tx:
                    litres = rng.gamma(2.2, st["mean_tx"] / 2.2, n_tx).clip(4, 160).round(2)
                    avail = max(st["stock"] - st["dead"], 0.0)
                    while litres.sum() > avail and len(litres):
                        litres = litres[:-1]
            actual = float(litres.sum())

            # --- pompa kalibrasyon / decimal: kaydedilen litre çarpanı
            scale = 1.0
            if w_active and w_active[0] == "pompa_kalibrasyon" and actual > 0:
                scale = 1 + w_active[3]; cat, label = "pompa_kalibrasyon", 1
            dec_now = (key == decimal_tank and day == decimal_day)

            # --- sızıntılar
            leak = 0.0
            if w_active and w_active[0] == "statik_sizinti":
                leak = w_active[3] * rng.uniform(0.8, 1.2); cat, label = "statik_sizinti", 1
            elif w_active and w_active[0] == "dinamik_sizinti" and actual > 0:
                leak = actual * w_active[3]; cat, label = "dinamik_sizinti", 1

            # --- su faktörü
            st["su"] += max(rng.normal(0.0015, 0.001), 0)
            if key == su_tank and su_d0 <= day < su_d1:
                st["su"] += rng.uniform(0.03, 0.06)
                st["stock"] += rng.uniform(0.5, 2.0)
                cat, label = "su_faktoru", 1

            # --- mapping hatası
            phys_out = actual
            if map_cfg and ist == map_cfg[0] and map_cfg[3] <= day < map_cfg[4]:
                if tno == map_cfg[1] and actual > 0:
                    frac = 1.0 / max(len(nozzles[key]), 2)
                    m = actual * frac
                    phys_out = actual - m
                    misroute[map_cfg[2]] = misroute.get(map_cfg[2], 0) + m
                    cat, label = "mapping_hatasi", 1
                if tno == map_cfg[2] and misroute.get(tno, 0) > 0:
                    st["stock"] -= misroute.pop(tno)
                    cat, label = "mapping_hatasi", 1

            # --- manifold transfer
            ev = transfer.get((ist, p))
            if ev and tno in (ev[0], ev[1]):
                st["stock"] += (-ev[2] if tno == ev[0] else ev[2])
                cat, label = "manifold_yakit_gecisi", 1

            # --- dinamik ayarlamalar (bölmeli vb.)
            for delta, c in dyn.pop((tno, p), []):
                st["stock"] += delta
                cat, label = c, 1

            # --- nokta olayları
            extra_metered = 0.0
            test_litres = []
            for typ, X in point.get((ist, tno, p), []):
                if typ == "yakit_cekimi":
                    st["stock"] -= X; cat, label = "yakit_cekimi", 1
                elif typ == "pompaci":
                    st["stock"] -= X; cat, label = "pompaci_manipulasyonu", 1
                elif typ == "test_satisi":
                    test_litres = [round(v, 2) for v in
                                   rng.gamma(3, X / 6, 2).clip(10, 60)]
                    extra_metered = sum(test_litres)          # yakıt geri döküldü
                    cat, label = "test_satisi", 1
                elif typ == "hayali_dolum":
                    dolum += X
                    delivery_id += 1
                    del_rows.append(dict(
                        dolum_id=delivery_id,
                        dolum_baslangic=t0 + pd.Timedelta(minutes=2),
                        dolum_bitis=t0 + pd.Timedelta(minutes=20),
                        istasyon_kodu=ist, tank_no=tno, akaryakit_turu=st["urun_ad"],
                        dolum_oncesi_hacim=round(st["stock"], 2),
                        dolum_sonrasi_hacim=round(st["stock"], 2),
                        dolum_brut=round(X, 2), dolum_net=round(X, 2),
                        sicaklik=np.nan, merkeze_gelis_tarihi=pd.NaT))
                    cat, label = "hayali_dolum", 1
                elif typ == "algilanmayan_dolum":
                    st["stock"] += min(X, st["kap"] * 0.9 - st["stock"])
                    cat, label = "algilanmayan_dolum", 1
                elif typ == "gsd_fiziksel":
                    st["stock"] += X; cat, label = "gun_sonu_dolum", 1
                elif typ == "gsd_kayit":
                    dolum += X
                    delivery_id += 1
                    del_rows.append(dict(
                        dolum_id=delivery_id,
                        dolum_baslangic=t0 - pd.Timedelta(minutes=15),
                        dolum_bitis=t0 + pd.Timedelta(minutes=10),
                        istasyon_kodu=ist, tank_no=tno, akaryakit_turu=st["urun_ad"],
                        dolum_oncesi_hacim=round(st["stock"] - X, 2),
                        dolum_sonrasi_hacim=round(st["stock"], 2),
                        dolum_brut=round(X * 1.002, 2), dolum_net=round(X, 2),
                        sicaklik=round(st["temp"], 2), merkeze_gelis_tarihi=t1))
                    cat, label = "gun_sonu_dolum", 1

            # --- ölçüm gürültüsü ve fiziksel stok kapanışı
            st["stock"] += rng.normal(0, 0.6) - phys_out - leak
            st["stock"] = max(st["stock"], st["dead"] * 0.4)

            # --- tekil satışları yaz
            metered_exact = 0.0
            pr = price[st["urun"]][day]
            if len(litres):
                secs = np.sort(rng.integers(0, 1799, len(litres)))
                for li, s in zip(litres, secs):
                    pn, tabn = nozzles[key][int(rng.integers(0, len(nozzles[key])))]
                    rec = round(float(li) * scale * (10 if dec_now else 1), 2)
                    unmapped = rng.random() < 0.011
                    tx_rows.append(dict(
                        satis_zamani=t0 + pd.Timedelta(seconds=int(s)),
                        istasyon_kodu=ist, tank_no=np.nan if unmapped else tno,
                        pompa_no=pn, tabanca_no=tabn, akaryakit_turu=st["urun_ad"],
                        satis_tipi=rng.choice(["Pompaci", "Kendin Sec"], p=[0.85, 0.15]),
                        litre=rec, birim_fiyat=round(pr, 2),
                        tutar=round(float(li) * pr, 2), test_satisi=0))
                    metered_exact += rec
            for li in test_litres:
                pn, tabn = nozzles[key][0]
                tx_rows.append(dict(
                    satis_zamani=t0 + pd.Timedelta(seconds=int(rng.integers(0, 1799))),
                    istasyon_kodu=ist, tank_no=tno, pompa_no=pn, tabanca_no=tabn,
                    akaryakit_turu=st["urun_ad"], satis_tipi="Pompaci",
                    litre=li, birim_fiyat=round(pr, 2), tutar=round(li * pr, 2),
                    test_satisi=1))
                metered_exact += li
            metered_exact = round(metered_exact, 2)
            if dec_now and metered_exact > 0:
                cat, label = "pompa_decimal", 1

            # --- raporlanan (reported) stok
            if stuck:
                if st["frozen"] is None:
                    st["frozen"] = st["rep_prev"]
                rep_end = st["frozen"]
                cat, label = "samandira_takilmasi", 1
                special = True
            elif atg or probe0:
                rep_end = 0.0
                cat, label = ("atg_arizasi", 1) if atg else ("probe_arizasi", 1)
                special = True
            else:
                if st["frozen"] is not None:        # şamandıra çözüldü
                    st["frozen"] = None
                    cat, label = "samandira_takilmasi", 1
                elif st["was_special"]:             # arıza bitti, gerçek değere dönüş
                    cat, label = ("atg_arizasi", 1) if key[0] == ATG_IST else ("probe_arizasi", 1)
                rep_end = round(st["stock"], 2)
                special = False

            rep_start = st["rep_prev"]
            kayip = round(rep_end - rep_start - round(dolum, 2) + metered_exact, 2)
            # yalnız dolum şoku gibi ani, büyük genleşmeler etiketlenir
            if cat == "normal" and abs(temp_eff) > 30:
                cat, label = "sicaklik_degisimi", 1

            temp_out = (0.0 if (atg or probe0)
                        else (np.nan if (key, day) in temp_fail else round(st["temp"], 2)))
            seviye0 = rep_start / st["lpc"]
            seviye1 = rep_end / st["lpc"]

            skip_row = elek or tkm_drop
            if not skip_row:
                st["cum"] += kayip
                ue1t_rows.append(dict(
                    saat_1=t0, saat_2=t1, istasyon_kodu=ist, tank_no=tno,
                    akaryakit_turu=st["urun_ad"], sicaklik=temp_out,
                    donem_basi_stok=rep_start, baslangic_seviyesi_cm=round(seviye0, 4),
                    tanka_dolum=round(dolum, 2), donem_sonu_stok=rep_end,
                    bitis_seviyesi_cm=round(seviye1, 4),
                    tank_seviyesi_azalma_miktari=round(rep_start - rep_end + dolum, 2),
                    pompa_satis=metered_exact, kayip_kazanc=kayip,
                    kumulatif_kayip_kazanc=round(st["cum"], 2),
                    oran=round(100 * kayip / metered_exact, 2) if metered_exact > 1 else 0.0,
                    anomali_etiketi=label, anomali_kategorisi=cat))
                wf = (key, day) in water_fail
                net = (rep_end * (1 - beta * (st["temp"] - 15))
                       if not np.isnan(temp_out) and not (atg or probe0) else
                       (0.0 if (atg or probe0) else np.nan))
                gelis = t1 + pd.Timedelta(minutes=float(np.clip(rng.normal(6, 3), 1, 25)))
                if ist == ROUTER_IST and day == router_day:
                    gelis = (START + pd.Timedelta(days=day + 1)
                             + pd.Timedelta(hours=float(rng.uniform(2, 8))))
                inv_rows.append(dict(
                    envanter_tarihi=t1, istasyon_kodu=ist, tank_no=tno,
                    akaryakit_turu=st["urun_ad"],
                    urun_seviyesi_cm=round(seviye1, 4),
                    urun_miktari_brut=rep_end,
                    urun_miktari_net=round(net, 2) if not np.isnan(net) else np.nan,
                    su_seviyesi_cm=np.nan if wf else (0.0 if (atg or probe0)
                                                      else round(st["su"], 3)),
                    su_miktari=np.nan if wf else (0.0 if (atg or probe0)
                                                  else round(st["su"] * st["lpc"] * 0.9, 2)),
                    sicaklik=temp_out, merkeze_gelis_tarihi=gelis))
                st["rep_prev"] = rep_end
            st["was_special"] = special

ue1t = pd.DataFrame(ue1t_rows)
inv = pd.DataFrame(inv_rows)
tx = pd.DataFrame(tx_rows).sort_values("satis_zamani").reset_index(drop=True)
deliveries = pd.DataFrame(del_rows)

# ------------------------------------------------------------------ daily
ue1t["_tarih"] = ue1t.saat_1.dt.normalize()

def day_cat(s):
    nn = s[s != "normal"]
    return nn.mode().iloc[0] if len(nn) else "normal"

daily = ue1t.groupby(["istasyon_kodu", "tank_no", "_tarih"], sort=True).agg(
    akaryakit_turu=("akaryakit_turu", "first"),
    acilis=("donem_basi_stok", "first"), kapanis=("donem_sonu_stok", "last"),
    dolum=("tanka_dolum", "sum"), satis=("pompa_satis", "sum"),
    anomali_etiketi=("anomali_etiketi", "max"),
    anomali_kategorisi=("anomali_kategorisi", day_cat)).reset_index()
daily = daily.rename(columns={"_tarih": "tarih"})
daily["azalma_miktari"] = (daily.acilis - daily.kapanis + daily.dolum).round(2)
daily["fark"] = (daily.satis - daily.azalma_miktari).round(2)
daily["oran"] = np.where(daily.satis > 1, (100 * daily.fark / daily.satis).round(2), 0.0)
daily["sel"] = (0.005 * daily.satis + 24).round(2)
daily["alarm"] = (daily.fark.abs() > daily.sel).astype(int)
# elektrik arızası günü: detay satırları eksik, gün etiketlenir
elek_day = elek_p0 // PPD
me = (daily.istasyon_kodu == ELEK_IST) & (daily.tarih == START + pd.Timedelta(days=elek_day))
daily.loc[me, ["anomali_etiketi", "anomali_kategorisi"]] = [1, "elektrik_arizasi"]
daily = daily[["tarih", "istasyon_kodu", "tank_no", "akaryakit_turu", "acilis", "kapanis",
               "dolum", "azalma_miktari", "satis", "fark", "oran", "sel", "alarm",
               "anomali_etiketi", "anomali_kategorisi"]]
ue1t = ue1t.drop(columns=["_tarih"])

# ------------------------------------------------------------------ ek null enjeksiyonu (EDA / imputation pratiği)
def inject_extra_nulls(tx, ue1t, inv, deliveries, mapping):
    """Senaryo null'larına ek rastgele eksikler — kümelenmiş ve dağınık."""
    n = len(tx)
    for idx in rng.choice(n, size=int(n * 0.004), replace=False):
        tx.at[idx, "birim_fiyat"] = np.nan
    for idx in rng.choice(n, size=int(n * 0.002), replace=False):
        tx.at[idx, "litre"] = np.nan
    for idx in rng.choice(len(ue1t), size=int(len(ue1t) * 0.003), replace=False):
        ue1t.at[idx, "baslangic_seviyesi_cm"] = np.nan
    for idx in rng.choice(len(inv), size=int(len(inv) * 0.002), replace=False):
        inv.at[idx, "merkeze_gelis_tarihi"] = pd.NaT
    for idx in rng.choice(len(deliveries), size=int(len(deliveries) * 0.08), replace=False):
        deliveries.at[idx, "dolum_oncesi_hacim"] = np.nan
    if len(mapping) > 2:
        for idx in rng.choice(len(mapping), size=2, replace=False):
            mapping.at[idx, "manifold_grup_no"] = np.nan

inject_extra_nulls(tx, ue1t, inv, deliveries, mapping)

# ------------------------------------------------------------------ ham operasyonel tablolar vs ground truth (ML aşaması için)
GT = OUT / "ground_truth"
GT.mkdir(exist_ok=True)

gt_ue1t = ue1t[["saat_1", "saat_2", "istasyon_kodu", "tank_no",
                "anomali_etiketi", "anomali_kategorisi"]].copy()
gt_daily = daily[["tarih", "istasyon_kodu", "tank_no",
                  "anomali_etiketi", "anomali_kategorisi"]].copy()

ue1t_ops = ue1t.drop(columns=["anomali_etiketi", "anomali_kategorisi"])
daily_ops = daily.drop(columns=["anomali_etiketi", "anomali_kategorisi"])

# ------------------------------------------------------------------ kaydet
for n, df in [("stations", stations), ("tanks", tanks), ("mapping", mapping),
              ("transactions", tx), ("deliveries", deliveries),
              ("inventory_30min", inv), ("ue1t_30min", ue1t_ops), ("daily", daily_ops)]:
    df.to_csv(OUT / f"{n}.csv", index=False)
gt_ue1t.to_csv(GT / "labels_30min.csv", index=False)
gt_daily.to_csv(GT / "labels_daily.csv", index=False)

# ------------------------------------------------------------------ doğrulama
print("=" * 72)
print("TABLOLAR")
for n, df in [("stations", stations), ("tanks", tanks), ("mapping", mapping),
              ("transactions", tx), ("deliveries", deliveries),
              ("inventory_30min", inv), ("ue1t_30min", ue1t_ops), ("daily", daily_ops)]:
    print(f"  {n:<16} {len(df):>7} satır x {df.shape[1]} kolon")
print(f"  {'ground_truth/labels_30min':<16} {len(gt_ue1t):>7} satır (ML doğrulama — EDA'da kullanma)")
print(f"  {'ground_truth/labels_daily':<16} {len(gt_daily):>7} satır (ML doğrulama — EDA'da kullanma)")

res = ue1t.donem_sonu_stok - (ue1t.donem_basi_stok + ue1t.tanka_dolum
                              - ue1t.pompa_satis + ue1t.kayip_kazanc)
print(f"\nue1t mutabakat denklemi  max|res| = {res.abs().max():.4f}")

n = ue1t[ue1t.anomali_kategorisi == "normal"]
print(f"normal kayip: mean={n.kayip_kazanc.mean():.3f} std={n.kayip_kazanc.std():.3f}")
dn = daily[daily.anomali_kategorisi == "normal"]
print(f"daily(normal) fark: median={dn.fark.median():.2f}  P95|fark|={dn.fark.abs().quantile(.95):.1f}")
print(f"daily alarm oranı = {daily.alarm.mean():.3f} | anomali gün oranı = {daily.anomali_etiketi.mean():.3f}")
print(f"30dk anomali oranı = {(ue1t.anomali_etiketi == 1).mean():.4f}")
print("\nKATEGORİ KIRILIMI (30dk satır):")
print(ue1t[ue1t.anomali_etiketi == 1].anomali_kategorisi.value_counts().to_string())
print("\nNULL / EKSİK:")
print(f"  ue1t sicaklik NaN={ue1t.sicaklik.isna().sum()}  inv net NaN={inv.urun_miktari_net.isna().sum()}"
      f"  inv su NaN={inv.su_seviyesi_cm.isna().sum()}")
print(f"  tx tank_no NaN={tx.tank_no.isna().sum()}  deliveries sicaklik NaN={deliveries.sicaklik.isna().sum()}"
      f"  merkez NaN={deliveries.merkeze_gelis_tarihi.isna().sum()}")
print(f"  eksik 30dk satırı = {N_PERIODS * len(tanks) - len(ue1t)}"
      f" (elektrik={elek_len}x{(tanks.istasyon_kodu == ELEK_IST).sum()}tank + tankomat)")
print(f"\nSENARYO YERLERİ (kontrol için):")
print(f"  elektrik: {ELEK_IST} p{elek_p0}-{elek_p0+elek_len} | atg: {ATG_IST} p{atg_p0}-{atg_p0+atg_len}")
print(f"  router: {ROUTER_IST} gün{router_day} | tankomat: {sorted(tankomat_days)}")
print(f"  probe0: {probe_tank} p{probe_p0} | stuck: {stuck_tank} gün{stuck_d0}-{stuck_d1}")
print(f"  su: {su_tank} gün{su_d0}-{su_d1} | decimal: {decimal_tank} gün{decimal_day}")
print(f"  mapping: {map_cfg} | bölmeli: {BOLME}")
print("\nOK — data/ klasörüne yazıldı.")
