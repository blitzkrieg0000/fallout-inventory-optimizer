# Fallout 76 Stash Optimizer - Formül Kılavuzu

## İçindekiler

1. [Temel Kavramlar](#1-temel-kavramlar)
2. [Bileşik Önem Skoru](#2-bileşik-önem-skoru)
3. [Faz A: Kategori İçi Cascade](#3-faz-a-kategori-içi-cascade)
4. [Faz B: Global Havuz Dağıtımı](#4-faz-b-global-havuz-dağıtımı)
5. [Preferred Min/Max Penceresi](#5-preferred-minmax-penceresi)
6. [SCIP Optimizasyonu](#6-scip-optimizasyonu)
7. [Asimetrik Sapma Ağırlıkları](#7-asimetrik-sapma-ağırlıkları)
8. [Epigraph Dönüşümü](#8-epigraph-dönüşümü)
9. [Kategori Give/Take Dengesi](#9-kategori-givetake-dengesi)
10. [Greedy Fallback](#10-greedy-fallback)
11. [Sabitler Tablosu](#11-sabitler-tablosu)

---

## 1. Temel Kavramlar

### Kapasite Hiyerarşisi

```
stash_capacity   toplam fiziksel stash kapasitesi
buffer           bilinçli boş bırakılan alan
usable           = stash_capacity - buffer
```

`usable` tüm hesaplamaların referans noktasıdır. Buffer asla kullanılmaz.

### Item Ağırlıkları

```
effective_weight = unit_weight * weight_modifier
total_weight     = effective_weight * optimal_quantity
```

`weight_modifier` varsayılan olarak 1.0'dır. Perks veya özel durumlar için kullanılır.

### type_count Mantığı

Aynı ağırlığa sahip benzer ürünleri tek satırda temsil etmek için kullanılır.

```
min_total = type_count * min_unit_quantity
max_total = type_count * max_unit_quantity
```

Örnek: `type_count=15`, `max_unit_quantity=3` ise toplam maksimum 45 adettir.

---

## 2. Bileşik Önem Skoru

Her item için `composite_importance` şu formülle hesaplanır:

```
base = 0.45 * category_importance + 0.55 * item_importance

eğer is_essential == True:
    composite = min(1.0, base + 0.15)
değilse:
    composite = base

composite = clamp(composite, 0.01, 1.0)
```

Kategori ağırlığı (%45) ile item ağırlığı (%55) birleştirilir. Essential bonus (+0.15) zorunlu itemların solver tarafından önceliklendirilmesini sağlar.

Bu skor tüm önem bazlı hesaplamalarda `composite_importance` olarak kullanılır.

---

## 3. Faz A: Kategori İçi Cascade

### Başlangıç Dağılımı

Her kategorinin nominal bütçesi önem skorlarına göre itemlara dağıtılır.

```
score_i     = composite_importance_i ^ IMPORTANCE_ALPHA
score_sum   = toplam(score_i)

alloc_i     = (score_i / score_sum) * cat_budget / effective_weight_i
```

`IMPORTANCE_ALPHA` üsteli önem farkını büyütür. 1.0 olursa lineer, 2.0 olursa karesel dağılım elde edilir.

`cat_budget = usable * allocation_ratio` formülünden gelir.

### Cascade Döngüsü

Başlangıç dağılımı min/max sınırlarını ihlal edebilir. Cascade bu ihlalleri düzeltir.

Her turda önce clamp kontrolü yapılır:

```
eğer alloc_i > max_total:
    pool += (alloc_i - max_total) * effective_weight_i
    alloc_i = max_total

eğer alloc_i < min_total:
    pool -= (min_total - alloc_i) * effective_weight_i
    alloc_i = min_total
```

Pozitif pool fazla ağırlık demektir (max'a çarpan itemlar verdi). Negatif pool eksik ağırlık demektir (min'e çarpan itemlar aldı).

Pool sıfıra yakınsamışsa (`abs(pool) < EPS`) döngü erken sonlanır.

### Havuz Yeniden Dağıtımı

Pool sıfır değilse adaylara dağıtılır:

```
eğer pool > 0:
    adaylar   = alloc_i < max_total olan itemlar
    cweight_i = score_i                     (önemliye daha çok ver)

eğer pool < 0:
    adaylar   = alloc_i > min_total olan itemlar
    cweight_i = 1.0 / (score_i + EPS)      (önemsizden daha çok al)

delta_i = (cweight_i / cweight_sum) * pool / effective_weight_i
alloc_i = clamp(alloc_i + delta_i, min_total, max_total)
```

Bu mantık önem sıralamasını korur: alan varken önemli itemlar büyür, alan yokken önemsiz itemlar küçülür.

### Global Havuza Aktarım

Cascade bittiğinde dağıtılamayan carry `global_pool`'a eklenir. Kategoriler arası dengeleme Faz B'de yapılır.

```
self.global_pool = carry
```

### Eşit Dağılım Garantisi

Bir kategorideki tüm itemlar özdeşse (aynı önem, aynı ağırlık, aynı sınırlar) basit ortalama uygulanır:

```
eğer tüm composite_importance eşit
    ve tüm effective_weight eşit
    ve tüm min_total eşit
    ve tüm max_total eşit:
        alloc_i = ortalama(alloc)  (tüm itemlar için)
```

---

## 4. Faz B: Global Havuz Dağıtımı

Kör scale (`usable / total_w`) yerine aynı cascade motoru (`_distribute`) tüm itemlara uygulanır.

```
başlangıç: alloc_i = mevcut ideal_quantity_i
seed_pool: self.global_pool
```

`_distribute` fonksiyonu Faz A ile aynı mantığı izler fakat bu sefer tüm kategorilerdeki itemlar birlikte değerlendirilir. Kategori sınırları yoktur; yalnızca item düzeyinde min/max sınırları geçerlidir.

### Hard-Clamp Fallback

CASCADE_PASSES turunda yakınsama sağlanamazsa küçük artık için kör scale devreye girer:

```
total_w  = toplam(alloc_i * effective_weight_i)
target_w = min(usable, total_w + carry)
scale    = target_w / total_w
alloc_i  = clamp(alloc_i * scale, min_total, max_total)
```

Bu fallback nadir tetiklenir ve artık miktar genellikle çok küçüktür.

---

## 5. Preferred Min/Max Penceresi

Preferred min/max yalnızca raporlama amaçlıdır, SCIP kısıtlarına girmez. Deviation (`d+`, `d-`) hesabında kullanılır.

```
pmin = ideal_quantity * (1.0 - 0.30)    (idealin yüzde 70'i)
pmax = ideal_quantity * (1.0 + 0.30)    (idealin yüzde 130'u)
```

Essential itemlar için alt sınır ek koruma alır:

```
hard_floor  = min_total * ESSENTIAL_HARD_FLOOR
pmin        = max(pmin, hard_floor)
```

Bu sayede ideal düşük çıksa bile essential item'ın preferred_min değeri min_total'ın belirli bir oranının altına inemez.

Non-essential itemlar için:

```
pmin = max(0.0, pmin)
```

Üst sınır her zaman max_total ile kırpılır:

```
preferred_max = min(max_total, ceil(pmax))
```

Deviation raporlaması:

```
deviation_positive = max(0, optimal_quantity - preferred_max)
deviation_negative = max(0, preferred_min - optimal_quantity)
```

---

## 6. SCIP Optimizasyonu

### Değişkenler

```
q_i            miktar değişkeni (tam sayı veya sürekli)
d_neg_i        idealin altına düşme miktarı (>= 0)
d_pos_i        idealin üstüne çıkma miktarı (>= 0)
below_i        min_total altına düşme miktarı (>= 0)
give_c         kategorinin bütçe verme değişkeni (>= 0)
take_c         kategorinin bütçe alma değişkeni (>= 0)
slack          boş kapasite değişkeni (>= 0)
t_neg_i        d_neg ceza epigraph değişkeni (>= 0)
t_pos_i        d_pos ceza epigraph değişkeni (>= 0)
```

### Kapasite Kısıtları (Hard)

```
cap_hi:   toplam(effective_weight_i * q_i) <= usable
cap_lo:   toplam(effective_weight_i * q_i) >= usable * FILL_LO
slack:    slack >= usable - toplam(effective_weight_i * q_i)
```

`FILL_LO = 0.95` ile kapasite en az yüzde 95 dolu olmalıdır.

### Item Sınırları

```
lb_i = ESSENTIAL_HARD_FLOOR * min_total     (essential item için)
lb_i = 0                                    (non-essential için)

ub_i = max_total                            (surplus yoksa)
ub_i = max_total * (1 + log_bonus)          (surplus varsa, sadece essential)

log_bonus = log(1 + SURPLUS_LOG_BASE * surplus / usable) * composite_importance
```

Surplus: tüm itemlar maksimuma alınsa bile usable dolmuyorsa oluşan fazlalıktır.

### Continuous Relaksasyon

Geniş aralıklı ve hafif itemlar için tam sayı yerine sürekli değişken kullanılır:

```
eğer (max_total - min_total) > CONT_RANGE_THR
    ve effective_weight < CONT_WEIGHT_THR:
        vtype = "C"   (sürekli, sonradan yuvarlanır)
değilse:
    vtype = "I"   (tam sayı)
```

Bu optimizasyon Branch-and-Bound ağacını küçülterek çözüm hızını artırır.

### Lineer Amaç Fonksiyonu

```
minimize:
    toplam(t_neg_i)                          sapma cezası (yokluk)
    + toplam(t_pos_i)                        sapma cezası (varlık)
    + toplam(floor_weight_i * below_i)       min altı ceza
    + W_EMPTY * slack                        boş alan cezası
    + toplam(give_cost_c)                    kategori bütçe verme
    + toplam(take_cost_c)                    kategori bütçe alma
```

Amaç fonksiyonu tamamen lineerdir. Quadratic ifadeler yalnızca kısıt tarafında bulunur.

---

## 7. Asimetrik Sapma Ağırlıkları

Her item için iki ayrı sapma ağırlığı hesaplanır:

```
raw_neg_i = W_DEV_NEG * composite_importance_i * (steal_factor_i + EPS)
raw_pos_i = W_DEV_POS * composite_importance_i / (steal_factor_i + EPS)

w_neg_i = (raw_neg_i / toplam(raw_neg)) * N
w_pos_i = (raw_pos_i / toplam(raw_pos)) * N
```

N toplam item sayısıdır. Normalizasyon tüm item ağırlıklarının toplamını N'e sabitler; böylece tek bir item tüm ceza bütçesini tüketemez.

### steal_factor Etkisi

```
steal_factor yüksek ise:
    w_neg artar   altına düşmesi pahalı   solver onu kesmekten kaçınır
    w_pos azalır  üstüne çıkması ucuz     alan varsa önce bu item şişer

steal_factor düşük ise:
    w_neg azalır  altına düşmesi ucuz     alan sıkışınca bu item kesilir
    w_pos artar   üstüne çıkması pahalı   fazla alan bu iteme gitmez
```

Yüksek steal_factor'lü item hem agresif büyür hem de kesilmeye direnç gösterir.

---

## 8. Epigraph Dönüşümü

PySCIPOpt amaç fonksiyonunda quadratic ifadeye izin vermez. Epigraph tekniği bunu kısıt tarafına taşır.

### Problem

```
minimize  w * d^2          bu doğrudan yazılamaz
```

### Çözüm

```
t adlı yardımcı değişken ekle (t >= 0)

kısıt:  t >= w * d^2       quadratic KISIT olarak yazılabilir
amaç:   minimize t          tamamen lineer
```

Solver `t`'yi minimize etmek için `d`'yi küçültmek zorundadır. Optimumda `t = w * d^2` eşitliği kendiliğinden sağlanır.

### d_neg ve d_pos Tanımı

```
kısıt: d_neg >= ideal - q    (d_neg lb=0 olduğundan max(0, ideal-q) davranışı)
kısıt: d_pos >= q - ideal    (d_pos lb=0 olduğundan max(0, q-ideal) davranışı)
```

```
q < ideal:   d_neg = ideal - q,  d_pos = 0
q = ideal:   d_neg = 0,          d_pos = 0
q > ideal:   d_neg = 0,          d_pos = q - ideal
```

İkisi aynı anda pozitif olamaz; biri her zaman sıfırdır.

### Epigraph Kısıtları

```
t_neg >= w_neg * d_neg^2    (quadratic kısıt, solver kabul eder)
t_pos >= w_pos * d_pos^2    (quadratic kısıt, solver kabul eder)
```

Amaç fonksiyonuna yalnızca `t_neg + t_pos` girer, tamamen lineerdir.

### Min Altı Ceza

```
kısıt: below >= min_total - q

floor_weight = W_FLOOR * 3.0    (essential item için daha sert)
floor_weight = W_FLOOR * 1.0    (non-essential item için)

amaç katkısı: floor_weight * below   (lineer)
```

---

## 9. Kategori Give/Take Dengesi

Kategoriler nominal bütçelerinden fazla veya eksik kullanabilir. Bu esneklik `give` ve `take` değişkenleriyle modellenir.

### Değişken Sınırları

```
give_c sınırı: [0,  T_c]
take_c sınırı: [0,  MAX_TAKE_FRACTION * T_c]

T_c = nominal_budget(usable)  (kategorinin nominal bütçesi)
MAX_TAKE_FRACTION = 0.60      (bir kategori bütçesinin en fazla yüzde 60'ını alabilir)
```

### Denge Kısıtı (Hard)

```
toplam(give_c) == toplam(take_c)
```

Sistemde toplam ağırlık korunur: verilen kadar alınır.

### Ceza Fonksiyonu (Asimetrik)

```
give_cost_c = W_CAT_GIVE * (1 - importance_c)^2 * give_c
take_cost_c = W_CAT_TAKE * (1 - importance_c)   * take_c
```

Önemsiz kategoriler ucuza verir (`(1-imp)^2` büyük). Önemli kategoriler ucuza alır (`(1-imp)` küçük). Böylece bütçe transferi doğal olarak önem hiyerarşisini izler.

---

## 10. Greedy Fallback

SCIP kurulu değilse veya `--strategy greedy` seçilirse devreye girer.

### Azaltma (Reduce)

Toplam ağırlık usable'ı aşarsa her item için steal önceliği hesaplanır:

```
steal_priority_i = (1 - composite_importance_i) * steal_factor_i
```

Yüksek öncelikli itemlar önce kesilir. Her turda kesim orantılı dağıtılır:

```
cut_w_i = over_weight * (steal_priority_i / toplam(steal_priority))
cut_u_i = min(q_i - floor_i, ceil(cut_w_i / effective_weight_i))
q_i    -= cut_u_i
```

Hard floor koruması:

```
floor_i = max(1, floor(min_total * ESSENTIAL_HARD_FLOOR))   (essential)
floor_i = 1                                                  (non-essential, 0'a inemez)
```

### Genişleme (Expand)

Toplam ağırlık `usable * FILL_LO` altındaysa itemlar büyütülür. Sıralama kriteri:

```
sıralama_skoru = composite_importance * steal_factor   (azalan)
```

Her item için:

```
room = usable - toplam_kullanılan_ağırlık
add  = min(ub_i - q_i, floor(room / effective_weight_i))
q_i += add
```

Surplus varsa essential itemlar için logaritmik üst sınır bonusu uygulanır:

```
log_bonus = log(1 + SURPLUS_LOG_BASE * surplus / usable) * composite_importance
ub_i      = max_total + floor(log_bonus * max_total)
```

---

## 11. Sabitler Tablosu

| Sabit | Varsayılan | Açıklama |
|---|---|---|
| `IMPORTANCE_ALPHA` | 1.6 | Önem üsteli (1=lineer, 2=karesel) |
| `CASCADE_PASSES` | 80 | Maksimum cascade turu |
| `FILL_LO` | 0.95 | Minimum doluluk oranı |
| `ESSENTIAL_HARD_FLOOR` | 0.80 | Essential min koruması çarpanı |
| `SURPLUS_LOG_BASE` | 2.4 | Surplus bonus logaritma tabanı |
| `MAX_TAKE_FRACTION` | 0.60 | Kategori maksimum bütçe alma oranı |
| `CONT_RANGE_THR` | 500 | Continuous relaksasyon aralık eşiği |
| `CONT_WEIGHT_THR` | 0.05 | Continuous relaksasyon ağırlık eşiği |
| `W_DEV_NEG` | 10.0 | Yokluk sapma ceza ağırlığı |
| `W_DEV_POS` | 1.0 | Varlık sapma ceza ağırlığı |
| `W_FLOOR` | 12.0 | Min altı ceza taban ağırlığı |
| `W_EMPTY` | 50.0 | Boş kapasite ceza ağırlığı |
| `W_CAT_GIVE` | 4.0 | Kategori bütçe verme ceza ağırlığı |
| `W_CAT_TAKE` | 2.0 | Kategori bütçe alma ceza ağırlığı |

---

## Akış Özeti

```
inventory.json
      |
      v
InventoryLoader.load()
      |
      v
IdealCalculator.compute()
      |
      +-> _cascade()         her kategori için (Faz A)
      |       |
      |       +-> _distribute()   havuz dağıtım motoru
      |
      +-> _global_rescale()  global havuz dağıtımı (Faz B)
              |
              +-> _distribute()   aynı motor, tüm itemlar
      |
      v
SCIPOptimizer.optimize()  veya  GreedyFallback.optimize()
      |
      v
ReportRenderer.render()   konsol tablosu
      |
      v
JSONExporter.export()     optimization_result.json
```
