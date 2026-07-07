# Turpak – Staj Çalışmaları

Bu depo, **Turpak** stajım süresince geliştirdiğim çalışmaları, prototipleri ve tamamladığım taskları içerir. Amaç, ATG / akaryakıt otomasyon sistemleri üzerine öğrendiklerimi kod örnekleriyle belgelemektir.

**Stajyer:** Ali Edip Mangtay

---

## İçindekiler

| Dosya | Açıklama |
|-------|----------|
| [`strapping_table_model.py`](strapping_table_model.py) | Yatay silindirik akaryakıt tankı için seviye → hacim (cetvel/strapping table) interpolasyon modeli |

---

## Projeler

### 1. Strapping Table (Cetvel) Interpolasyon Modeli

Yatay silindirik bir akaryakıt tankında **seviye ile hacim arasındaki ilişki doğrusal değildir**: tankın ortasında 1 cm'lik seviye değişimi çok daha fazla litreye karşılık gelirken, alt ve üst kısımlarda daha az litreye karşılık gelir.

ATG / otomasyon sistemleri bu yüzden bir **cetvel tablosu (strapping table)** tutar: ayrık kalibrasyon noktaları (seviye → hacim). Ara seviyeler ise en yakın iki nokta arasında **doğrusal interpolasyon** ile tahmin edilir.

Bu script:

1. Tank geometrisinden (dairesel kesit formülü) **tam (exact)** seviye → hacim eğrisini hesaplar,
2. Bu eğriyi sabit adımlarla örnekleyerek bir **cetvel tablosu** oluşturur,
3. Sistemin kullandığı **doğrusal interpolasyonu** uygular,
4. İnterpolasyon (tahmin) **hatasını** ölçer,
5. Eşit seviye değişimlerinin neden eşit hacim değişimi vermediğini gösterir.

#### Örnek Çıktı

```
Tank: R=1.2 m, L=8.0 m, full volume = 36,191 L
10 cm table ->  25 rows | max |error| =   75.4 L
25 cm table ->  10 rows | max |error| =  941.8 L
  1 cm change at level 0.10 m  ->    78.5 L
  1 cm change at level 1.20 m  ->   192.0 L
  1 cm change at level 2.30 m  ->    74.9 L
```

Script ayrıca sonuçları `strapping_table_model.png` olarak kaydeder:
- Seviye → hacim eğrisinin doğrusal olmadığı
- Aynı Δseviye'nin farklı Δhacim verdiği
- Tablo çözünürlüğüne göre interpolasyon hatası

---

## Kurulum ve Çalıştırma

Gereksinimler:

```bash
pip install numpy matplotlib
```

Çalıştırma:

```bash
python strapping_table_model.py
```
