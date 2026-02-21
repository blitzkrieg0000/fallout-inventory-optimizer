"""
Fallout 76 – Stash Inventory Optimizer  v1.0
=============================================

Gereksinimler
-------------
• JSON'dan oku → optimize et → tablo + JSON çıktısı ver.
• Kapasite: stash_capacity − buffer = usable. Toplam ağırlık ≤ usable.
• Her item: quantity ∈ [0, max_total], essential ise hard_lb ≥ ESSENTIAL_FLOOR × min_total.
• İdeal hedef: importance-weighted cascade (IdealCalculator).
• Tek geçişli MIQCP (SCIP): normalize edilmiş sapma penaltisi + logaritmik zemin.
• Smooth spreading: deviation ağırlıkları normalize edilir → bir itemın önemi
  artınca tüm diğer itemlardan orantılı pay alır, tek bir item'dan almaz.
• Logaritmik zemin: α×(below²+below) → items 0'a yapışmaz.
• Surplus: Σmax_weight < usable ise önemli itemlar max üstüne çıkabilir.
• Give/Take: kategori bazlı bütçe akışı soft penalti ile izlenir.
• Kapasite band: Σ w·q ∈ [usable×FILL_TOL, usable]. Her zaman feasible.
• allocation_ratio toplamı 1 olmak zorunda değil; her kategori kendi
  oransal bütçesiyle çalışır.
• type_count: optimal_quantity = type_count × qty_per_type.
  max_total = type_count × max_unit_quantity, vs.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Opsiyonel bağımlılıklar
try:
    from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False
    print("[WARN] pyscipopt bulunamadı – Greedy fallback kullanılacak.", file=sys.stderr)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

_EPS = 1e-9

#! Configuration
# IdealCalculator
IMPORTANCE_ALPHA: float = 1.6   # önem üstelleştirme (1=lineer, 2=karesel)
CASCADE_PASSES:   int   = 40    # cascade yeniden dağıtım turları

# Kapasite doluluk bandı
FILL_TOLERANCE: float = 0.9500  # alt sınır: usable × bu oran ≥ Σw·q
#                                  (0.95 → %5 esneklik, her zaman feasible)

# Objective ağırlıkları
W_ITEM_DEV:    float = 1.0    # sapma penaltisi temel çarpanı
W_FLOOR_ESS:   float = 14.0   # essential altı zemin penaltisi (dik)
W_FLOOR_NON:   float = 2.5    # non-essential altı zemin penaltisi (yumuşak)
W_EMPTY:       float = 60.0   # boş kapasite penaltisi (her birim)
W_CAT_GIVE:    float = 5.0    # kategori bütçe verme penaltisi temel çarpanı
W_CAT_TAKE:    float = 3.0    # kategori bütçe alma penaltisi temel çarpanı
W_POOL_BAL:    float = 8.0    # Σgive≈Σtake soft denge penaltisi

# Sınırlar
ESSENTIAL_HARD_FLOOR: float = 0.80   # essential item'ın mutlak min'i = min_total × bu
SURPLUS_LOG_BASE:     float = 2.4    # surplus varsa max aşım faktörü
MAX_TAKE_FRACTION:    float = 0.60   # bir kategorinin alabileceği max ek bütçe oranı



#  VERİ MODELLERİ
@dataclass
class InventoryItem:
    """Envanterdeki tek bir item grubu."""

    name:                str
    category:            str
    type_count:          int        # aynı ağırlıktaki benzer ürün adedi
    unit_weight:         float      # bir birimin ağırlığı
    weight_modifier:     float      # ağırlık çarpanı (default 1.0)

    min_unit_quantity:   int        # bir tip için minimum miktar
    max_unit_quantity:   int        # bir tip için maksimum miktar

    category_importance: float      # kategorinin önemi [0,1]
    item_importance:     float      # itemın kendi önemi [0,1]
    is_essential:        bool       # essential ise zemin koruması daha güçlü
    steal_factor:        float      # 0→az, 1→çok – başkalarından ne kadar yer alır

    # IdealCalculator tarafından doldurulur
    ideal_quantity:    float = 0.0  # hedef miktar (float, kesirsiz hesap)
    baseline_quantity: int   = 0    # ideal'in yuvarlanmış hali

    # Solver tarafından doldurulur
    optimal_quantity:   int = 0
    preferred_min:      int = 0
    preferred_max:      int = 0
    deviation_positive: int = 0     # d+  → preferred_max üstünde
    deviation_negative: int = 0     # d-  → preferred_min altında

    def __post_init__(self) -> None:
        # Geçerlilik kontrolleri
        if self.type_count < 1:
            raise ValueError(f"[{self.name}] type_count ≥ 1 olmalı.")
        if self.min_unit_quantity < 0:
            raise ValueError(f"[{self.name}] min_unit_quantity ≥ 0 olmalı.")
        if self.max_unit_quantity < self.min_unit_quantity:
            raise ValueError(f"[{self.name}] max ≥ min olmalı.")

    @property
    def min_total(self) -> int:
        """Tüm type_count için toplam minimum (type_count × min_unit)."""
        return self.type_count * self.min_unit_quantity

    @property
    def max_total(self) -> int:
        """Tüm type_count için toplam maksimum (type_count × max_unit)."""
        return self.type_count * self.max_unit_quantity

    @property
    def effective_weight(self) -> float:
        """Bir birimin gerçek ağırlığı (unit_weight × modifier)."""
        return self.unit_weight * self.weight_modifier

    @property
    def total_weight(self) -> float:
        """Mevcut optimal_quantity için toplam ağırlık."""
        return self.effective_weight * self.optimal_quantity

    @property
    def qty_per_type(self) -> int:
        """Bir type için ortalama miktar (raporlama için)."""
        return self.optimal_quantity // self.type_count if self.type_count else 0

    @property
    def composite_importance(self) -> float:
        """
        Kategori ve item öneminin ağırlıklı birleşimi.
        essential ise bonus +0.15 (1.0'a kırpılır).
        """
        base = 0.45 * self.category_importance + 0.55 * self.item_importance
        if self.is_essential:
            base = min(1.0, base + 0.15)
        return round(min(1.0, max(0.01, base)), 6)

    @property
    def delta_from_baseline(self) -> int:
        return self.optimal_quantity - self.baseline_quantity


@dataclass
class Category:
    name:             str
    importance:       float
    allocation_ratio: float          # usable'ın bu oranı bu kategoriye ayrılır
    max_capacity:     Optional[float] = None  # isteğe bağlı hard üst sınır
    items:            list[InventoryItem] = field(default_factory=list)

    @property
    def allocated_weight(self) -> float:
        return sum(i.total_weight for i in self.items)

    @property
    def total_quantity(self) -> int:
        return sum(i.optimal_quantity for i in self.items)

    def nominal_budget(self, usable: float) -> float:
        """Bu kategorinin nominal bütçesi (ağırlık cinsinden)."""
        b = usable * self.allocation_ratio
        if self.max_capacity is not None:
            b = min(b, float(self.max_capacity))
        return b



# JSON Loader
class InventoryLoader:
    """inventory.json'u okuyup Category + InventoryItem listesi döner."""

    @staticmethod
    def load(path: str | Path) -> tuple[float, float, list[Category]]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        capacity: float = float(raw["stash_capacity"])
        buffer:   float = float(raw["buffer"])
        usable          = capacity - buffer

        if usable <= 0:
            raise ValueError("buffer ≥ stash_capacity! Kullanılabilir alan yok.")

        categories: list[Category] = []
        for cd in raw["categories"]:
            cat_imp   = float(cd["category_importance"])
            alloc_rat = float(cd["allocation_ratio"])

            items: list[InventoryItem] = []
            for ri in cd["items"]:
                item = InventoryItem(
                    name                = str(ri["name"]),
                    category            = str(cd["category"]),
                    type_count          = int(ri["type_count"]),
                    unit_weight         = float(ri["unit_weight"]),
                    weight_modifier     = float(ri.get("weight_modifier", 1.0)),
                    min_unit_quantity   = int(ri["min_unit_quantity"]),
                    max_unit_quantity   = int(ri["max_unit_quantity"]),
                    category_importance = cat_imp,
                    item_importance     = float(ri["item_importance"]),
                    is_essential        = bool(ri.get("is_essential", False)),
                    steal_factor        = float(ri.get("steal_factor", 0.5)),
                )
                items.append(item)

            categories.append(Category(
                name             = str(cd["category"]),
                importance       = cat_imp,
                allocation_ratio = alloc_rat,
                max_capacity     = cd.get("max_capacity"),
                items            = items,
            ))

        # allocation_ratio toplamı 1'den büyükse uyar ama hata verme
        total_ratio = sum(c.allocation_ratio for c in categories)
        if total_ratio > 1.0 + _EPS:
            print(f"[WARN] allocation_ratio toplamı {total_ratio:.3f} > 1.0 "
                  f"– kategoriler birbirleriyle rekabet eder.", file=sys.stderr)

        return capacity, buffer, categories



class IdealCalculator:
    """
    İki fazlı ideal miktar hesabı:

    Faz A – Kategori içi cascade:
      Her kategorinin nominal bütçesi içinde importance^alpha ağırlıklı
      orantılı dağıtım yapılır. Max'a çarpan itemlar fazlalığı havuza
      geri verir; min'in altında kalan itemlar havuzdan tamamlar.
      Bu süreç yakınsamaya kadar (CASCADE_PASSES tur) tekrarlanır.

    Faz B – Global rescale:
      Tüm ideallerin ağırlıklı toplamı usable'a normalize edilir.
      Böylece SCIP'e verilen hedef zaten tam kapasiteyi hedefler.
    """

    def __init__(self, usable: float, categories: list[Category]) -> None:
        self.usable     = usable
        self.categories = categories

    def compute(self) -> None:
        for cat in self.categories:
            self._cascade(cat)
        self._global_rescale()


    ##! --------------- Faz A --------------- !##
    def _cascade(self, cat: Category) -> None:
        items  = cat.items
        if not items:
            return

        budget = cat.nominal_budget(self.usable)
        scores = [i.composite_importance ** IMPORTANCE_ALPHA for i in items]
        total  = sum(scores) or 1.0

        # Başlangıç dağılımı (miktar cinsinden)
        alloc: list[float] = [
            (s / total) * budget / (i.effective_weight or _EPS)
            for s, i in zip(scores, items)
        ]

        # Cascade yeniden dağıtım
        for _ in range(CASCADE_PASSES):
            pool        = 0.0
            unsatisfied: list[int] = []

            for idx, item in enumerate(items):
                qty = alloc[idx]
                if qty > item.max_total:
                    pool     += (qty - item.max_total) * item.effective_weight
                    alloc[idx] = float(item.max_total)
                elif qty < item.min_total:
                    pool     -= (item.min_total - qty) * item.effective_weight
                    alloc[idx] = float(item.min_total)
                else:
                    unsatisfied.append(idx)

            if abs(pool) < _EPS or not unsatisfied:
                break

            unc_sum = sum(scores[j] for j in unsatisfied) or 1.0
            for j in unsatisfied:
                alloc[j] += (scores[j] / unc_sum) * pool / (items[j].effective_weight or _EPS)

        # Eşit önem/ağırlık durumunda tam eşitlik garantisi
        if (len({i.composite_importance for i in items}) == 1
                and len({i.effective_weight for i in items}) == 1
                and len({i.min_total for i in items}) == 1
                and len({i.max_total for i in items}) == 1
                and len(items) > 1):
            avg   = sum(alloc) / len(alloc)
            alloc = [avg] * len(alloc)

        # Item'lara yaz
        for item, ideal in zip(items, alloc):
            ideal = max(float(item.min_total), min(float(item.max_total), ideal))
            item.ideal_quantity    = ideal
            item.baseline_quantity = int(round(ideal))
            item.optimal_quantity  = item.baseline_quantity
            self._set_preferred(item)

    ##! --------------- Faz B --------------- !##
    def _global_rescale(self) -> None:
        all_items = [i for cat in self.categories for i in cat.items]
        total_w   = sum(i.ideal_quantity * i.effective_weight for i in all_items)
        if total_w < _EPS:
            return

        scale = self.usable / total_w
        for i in all_items:
            scaled             = max(float(i.min_total),
                                     min(float(i.max_total), i.ideal_quantity * scale))
            i.ideal_quantity    = scaled
            i.baseline_quantity = int(round(scaled))
            i.optimal_quantity  = i.baseline_quantity
            self._set_preferred(i)


    @staticmethod
    def _set_preferred(item: InventoryItem) -> None:
        """preferred_min / preferred_max: ideal ±%30."""
        flex = 0.30
        pmin = item.ideal_quantity * (1.0 - flex)
        pmax = item.ideal_quantity * (1.0 + flex)

        if item.is_essential:
            pmin = max(pmin, float(item.min_total) * ESSENTIAL_HARD_FLOOR)
        else:
            pmin = max(0.0, pmin)

        item.preferred_min = int(math.floor(pmin))
        item.preferred_max = int(math.ceil(min(float(item.max_total), pmax)))



class SCIPOptimizer:
    """
    MIQCP (Mixed-Integer Quadratically-Constrained) optimizasyon.

    Değişkenler
    -----------
    q_i   ∈ ℤ           : item miktarları
    give_c, take_c ≥ 0  : kategori bütçe akış değişkenleri (soft)
    slack ≥ 0            : boş kapasite (penaltili)
    below_i ≥ 0          : min_total altına iniş miktarı (penaltili)
    d_i (serbest)        : idealden sapma

    Hard kısıtlar
    -------------
    cap_hi  : Σ w·q ≤ usable
    cap_lo  : Σ w·q ≥ usable × FILL_TOLERANCE
    q_i ≥ hard_lb_i      (essential → ESSENTIAL_HARD_FLOOR × min_total)
    q_i ≤ hard_ub_i      (surplus varsa max aşımına izin)

    Soft amaç (minimize)
    ---------------------
    1) Normalize sapma  : norm_w_i × d_i²
       norm_w_i = (1/(imp·sf)) / Σ(1/(imp·sf)) × N
       → tüm itemlardan orantılı pay; tek itemdan çalınma yok

    2) Logaritmik zemin : α × (below² + below)
       → below→0 iken bile linear gradient var; 0'a yapışma yok

    3) Boş alan         : W_EMPTY × slack

    4) Kategori give/take (asimetrik):
       give cezası = W_CAT_GIVE × (1−imp)² × give²  (önemsiz → ucuz verir)
       take cezası = W_CAT_TAKE × (1−imp)  × take²  (önemli → ucuz alır)
       pool balance= W_POOL_BAL × (Σgive − Σtake)²  (soft denge)
    """

    def __init__(
        self,
        items:      list[InventoryItem],
        usable:     float,
        categories: list[Category],
    ) -> None:
        self.items      = items
        self.usable     = usable
        self.categories = categories

        # Kategori → item eşlemesi
        self._cat_items: dict[str, list[InventoryItem]] = {}
        for item in items:
            self._cat_items.setdefault(item.category, []).append(item)

        # Surplus (tüm max_total ağırlıklarının usable'dan küçük olduğu fazlalık)
        max_w          = sum(i.effective_weight * i.max_total for i in items)
        self._surplus  = max(0.0, usable - max_w)

        # Normalize edilmiş sapma ağırlıkları (smooth spreading)
        # önemli item → yüksek ceza → zor kesilir; yüksek sf → düşük ceza → kolay kesilir
        # Asimetrik sapma ağırlıkları
        # Yokluk (below ideal): imp × sf → yüksek sf → dik ceza → zor kesilir
        # Varlık  (above ideal): imp / sf → yüksek sf → düşük ceza → rahat genişler
        N = len(items)
        raw_neg = [W_ITEM_DEV * i.composite_importance * (i.steal_factor + _EPS)
                   for i in items]
        raw_pos = [W_ITEM_DEV * i.composite_importance / (i.steal_factor + _EPS)
                   for i in items]
        s_neg = sum(raw_neg) or 1.0
        s_pos = sum(raw_pos) or 1.0
        self._w_neg = [r / s_neg * N for r in raw_neg]  # cut protection
        self._w_pos = [r / s_pos * N for r in raw_pos]  # expand greed

        # Çözüm sonrası give/take değerleri
        self._give_vals: dict[str, float] = {}
        self._take_vals: dict[str, float] = {}

    # Sınır hesapları

    def _hard_lb(self, item: InventoryItem) -> int:
        """q_i için mutlak alt sınır."""
        if item.is_essential:
            return max(0, int(math.floor(item.min_total * ESSENTIAL_HARD_FLOOR)))
        return 0  # non-essential: 0'a kadar inebilir (penalty ile korunur)

    def _hard_ub(self, item: InventoryItem) -> int:
        """
        q_i için mutlak üst sınır.
        Surplus varsa önemli itemlar max_total üstüne logaritmik çıkabilir.
        """
        if self._surplus <= _EPS:
            return item.max_total
        frac  = math.log(1.0 + SURPLUS_LOG_BASE * self._surplus / (self.usable + _EPS))
        bonus = frac * item.composite_importance
        return int(math.ceil(item.max_total * (1.0 + bonus)))

    # Ana optimizasyon

    def optimize(self) -> None:
        if not HAS_SCIP:
            raise RuntimeError("pyscipopt kurulu değil.")

        m = Model("StashOptimizer_v1")
        m.hideOutput()
        m.setPresolve(SCIP_PARAMSETTING.AGGRESSIVE)
        m.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)

        # 1. Item miktar değişkenleri
        q: dict[int, object] = {}
        for item in self.items:
            iid    = id(item)
            q[iid] = m.addVar(
                vtype = "I",
                lb    = self._hard_lb(item),
                ub    = self._hard_ub(item),
                name  = f"q_{item.name[:18]}_{iid}",
            )

        # 2. Kapasite band kısıtları (HARD)
        total_w = quicksum(i.effective_weight * q[id(i)] for i in self.items)
        m.addCons(total_w <= self.usable,                    name="cap_hi")
        m.addCons(total_w >= self.usable * FILL_TOLERANCE,   name="cap_lo")

        # Boş alan değişkeni (penalty)
        slack = m.addVar(
            vtype = "C", lb = 0.0,
            ub    = self.usable * (1.0 - FILL_TOLERANCE) + 1.0,
            name  = "slack",
        )
        m.addCons(slack >= self.usable - total_w, name="slack_def")

        # 3. Kategori give/take değişkenleri (SOFT)
        give:   dict[str, object] = {}
        take:   dict[str, object] = {}
        t_give: dict[str, object] = {}
        t_take: dict[str, object] = {}

        for cat in self.categories:
            cid     = cat.name
            items_c = self._cat_items.get(cid, [])
            if not items_c:
                continue
            T_c = cat.nominal_budget(self.usable)
            imp = cat.importance

            give[cid] = m.addVar(vtype="C", lb=0.0, ub=T_c,
                                 name=f"give_{cid}")
            take[cid] = m.addVar(vtype="C", lb=0.0,
                                 ub=MAX_TAKE_FRACTION * T_c,
                                 name=f"take_{cid}")

            # Asimetrik penalty değişkenleri (quadratic constraint)
            gw = W_CAT_GIVE * ((1.0 - imp) ** 2 + _EPS)
            tg = m.addVar(vtype="C", lb=0.0, name=f"tg_{cid}")
            m.addCons(tg >= gw * give[cid] * give[cid], name=f"tg_qp_{cid}")
            t_give[cid] = tg

            tw = W_CAT_TAKE * ((1.0 - imp) + _EPS)
            tt = m.addVar(vtype="C", lb=0.0, name=f"tt_{cid}")
            m.addCons(tt >= tw * take[cid] * take[cid], name=f"tt_qp_{cid}")
            t_take[cid] = tt

        # Pool balance (soft): (Σgive − Σtake)² penalty
        t_pool = None
        cids   = list(give.keys())
        if len(cids) > 1:
            imbal  = m.addVar(vtype="C", lb=None, name="pool_imbal")
            m.addCons(
                imbal == quicksum(give[c] for c in cids)
                       - quicksum(take[c] for c in cids),
                name="pool_def",
            )
            t_pool = m.addVar(vtype="C", lb=0.0, name="t_pool")
            m.addCons(t_pool >= W_POOL_BAL * imbal * imbal, name="t_pool_qp")

        # 4. Asimetrik sapma penalty
        # d_neg = max(0, ideal-q) → yokluk: imp×sf ağırlığı (zor kesilir)
        # d_pos = max(0, q-ideal) → varlık:  imp/sf ağırlığı (rahat genişler)
        t_dev: dict[int, object] = {}
        for idx, item in enumerate(self.items):
            iid  = id(item)
            w_n  = self._w_neg[idx]   # below-ideal penalty weight
            w_p  = self._w_pos[idx]   # above-ideal penalty weight

            d_neg = m.addVar(vtype="C", lb=0.0, name=f"dneg_{iid}")
            d_pos = m.addVar(vtype="C", lb=0.0, name=f"dpos_{iid}")
            m.addCons(d_neg >= item.ideal_quantity - q[iid], name=f"dneg_def_{iid}")
            m.addCons(d_pos >= q[iid] - item.ideal_quantity, name=f"dpos_def_{iid}")

            td_neg = m.addVar(vtype="C", lb=0.0, name=f"tdneg_{iid}")
            td_pos = m.addVar(vtype="C", lb=0.0, name=f"tdpos_{iid}")
            m.addCons(td_neg >= w_n * d_neg * d_neg, name=f"tdneg_qp_{iid}")
            m.addCons(td_pos >= w_p * d_pos * d_pos, name=f"tdpos_qp_{iid}")

            # Toplam sapma penalty
            td = m.addVar(vtype="C", lb=0.0, name=f"td_{iid}")
            m.addCons(td >= td_neg + td_pos, name=f"td_sum_{iid}")
            t_dev[iid] = td

        # 5. Logaritmik zemin penaltisi: α×(below²+below)
        # below=0 → gradient=α (sıfırda bile itmek var → 0'a yapışmaz)
        t_floor: dict[int, object] = {}
        for item in self.items:
            if item.min_total <= 0:
                continue
            iid   = id(item)
            alpha = W_FLOOR_ESS if item.is_essential else W_FLOOR_NON

            below = m.addVar(vtype="C", lb=0.0, name=f"bel_{iid}")
            m.addCons(below >= item.min_total - q[iid], name=f"bel_def_{iid}")

            tf    = m.addVar(vtype="C", lb=0.0, name=f"tf_{iid}")
            m.addCons(tf >= alpha * (below * below + below), name=f"tf_qp_{iid}")
            t_floor[iid] = tf

        # 6. Amaç fonksiyonu
        obj = (
            quicksum(t_dev[iid]   for iid in t_dev)
          + quicksum(t_floor[iid] for iid in t_floor)
          + W_EMPTY * slack
          + quicksum(t_give[c]    for c in t_give)
          + quicksum(t_take[c]    for c in t_take)
        )
        if t_pool is not None:
            obj = obj + t_pool

        m.setObjective(obj, "minimize")

        #! Solve
        m.optimize()

        status = m.getStatus()
        if status not in ("optimal", "bestsolfound"):
            print(f"[WARN] SCIP status={status!r} – IdealCalculator baseline kullanılıyor.",
                  file=sys.stderr)
            self._finalise()
            return

        # Sonuçları yaz
        for item in self.items:
            val = m.getVal(q[id(item)])
            item.optimal_quantity = max(
                self._hard_lb(item),
                min(self._hard_ub(item), int(round(val))),
            )

        # Give/take değerlerini kaydet
        for c in give:
            self._give_vals[c] = m.getVal(give[c])
            self._take_vals[c] = m.getVal(take[c])

        self._finalise()

    def _finalise(self) -> None:
        for item in self.items:
            item.deviation_positive = max(0, item.optimal_quantity - item.preferred_max)
            item.deviation_negative = max(0, item.preferred_min  - item.optimal_quantity)

    def give_take_report(self) -> dict[str, dict[str, float]]:
        """Solver sonrası kategori bütçe akış raporu."""
        return {
            c: {"give": self._give_vals.get(c, 0.0),
                "take": self._take_vals.get(c, 0.0)}
            for c in self._give_vals
        }



# GREEDY FALLBACK  (SCIP yoksa)
class GreedyFallback:
    """
    Basit greedy algoritma.
    1. Baseline'dan başla.
    2. Kapasite aşılıyorsa → önemsizlerden kes.
    3. Kapasite altındaysa → önemlileri şişir.
    """
    def __init__(
        self,
        items:      list[InventoryItem],
        usable:     float,
        categories: list[Category],
    ) -> None:
        self.items      = items
        self.usable     = usable
        self.categories = categories

    def optimize(self) -> None:
        for item in self.items:
            item.optimal_quantity = item.baseline_quantity

        used = self._total_w()
        if used > self.usable:
            self._reduce()
        elif used < self.usable * FILL_TOLERANCE:
            self._expand()

        self._finalise()

    def _total_w(self) -> float:
        return sum(i.effective_weight * i.optimal_quantity for i in self.items)

    def _reduce(self) -> None:
        """Kapasiteyi aşan ağırlığı önemsizlerden ayır."""
        for _ in range(1000):
            over = self._total_w() - self.usable
            if over <= 0:
                break

            candidates = []
            for item in self.items:
                floor = (int(item.min_total * ESSENTIAL_HARD_FLOOR)
                         if item.is_essential else 0)
                room  = item.optimal_quantity - floor
                if room <= 0:
                    continue
                # Önemsiz itemdan daha çok kes (steal_weight yüksekse ucuz)
                sw = (
                    (1.0 - item.category_importance)
                  + (1.0 - item.composite_importance)
                  + (1.0 - item.steal_factor)  # yüksek sf → zor kurban → az kesilir
                ) * room * item.effective_weight
                candidates.append((item, sw, floor))

            if not candidates:
                break

            total_sw = sum(sw for _, sw, _ in candidates) or 1.0
            for item, sw, floor in candidates:
                cut_w = over * (sw / total_sw)
                cut_u = min(
                    item.optimal_quantity - floor,
                    max(0, int(math.ceil(cut_w / (item.effective_weight or 1)))),
                )
                item.optimal_quantity -= cut_u

    def _expand(self) -> None:
        """Boş alanı önemlileri şişirerek doldur."""
        max_w     = sum(i.effective_weight * i.max_total for i in self.items)
        glob_surp = max(0.0, self.usable - max_w)

        def ub(item: InventoryItem) -> int:
            if glob_surp <= _EPS:
                return item.max_total
            log_s = math.log(1.0 + SURPLUS_LOG_BASE * glob_surp / (self.usable + _EPS))
            return item.max_total + int(log_s * item.composite_importance * item.max_total)

        # Sıralama: category_importance, sonra composite_importance × steal_factor
        # → yüksek sf item artakalan alanı önce kapıyor (agresif genişleme)
        ranked = sorted(self.items,
                        key=lambda i: (i.category_importance,
                                       i.composite_importance * i.steal_factor),
                        reverse=True)
        for item in ranked:
            surplus = self.usable - self._total_w()
            if surplus <= 0:
                break
            gap = ub(item) - item.optimal_quantity
            if gap <= 0:
                continue
            add = min(gap, int(surplus / (item.effective_weight or 1)))
            item.optimal_quantity += add

    def _finalise(self) -> None:
        for item in self.items:
            item.deviation_positive = max(0, item.optimal_quantity - item.preferred_max)
            item.deviation_negative = max(0, item.preferred_min  - item.optimal_quantity)

    def give_take_report(self) -> dict[str, dict[str, float]]:
        return {}  # Greedy'de give/take yok


class ReportRenderer:
    """Sonuçları konsola tablo olarak basar."""

    _HDR   = ["Name", "TC", "Q/TC", "TotalQ", "Ideal",
               "ΔBase", "Range", "PMin", "PMax",
               "d+", "d-", "UnitW", "TotalW", "SF", "Imp", "Ess"]
    _COLW  = [26, 4, 6, 8, 8, 7, 15, 6, 6, 5, 5, 7, 8, 5, 5, 4]

    def render(
        self,
        categories:     list[Category],
        stash_capacity: float,
        buffer:         float,
        strategy:       str,
        give_take:      dict[str, dict[str, float]] | None = None,
    ) -> str:
        usable    = stash_capacity - buffer
        all_items = [i for cat in categories for i in cat.items]
        used      = sum(i.total_weight for i in all_items)
        gt        = give_take or {}

        W = 160
        lines: list[str] = []
        lines.append("═" * W)
        lines.append(" FALLOUT 76 – STASH OPTIMIZER v1.0 ".center(W, "═"))
        lines.append("═" * W)
        lines.append(
            f"  Strateji={strategy}  "
            f"Kapasite={stash_capacity:.0f}  Buffer={buffer:.0f}  "
            f"Kullanılabilir={usable:.0f}  "
            f"Kullanılan={used:.2f}  "
            f"Boş={usable - used:.2f}  "
            f"({100 * used / usable:.1f}%)"
        )
        lines.append("═" * W)

        rows: list[list[str]] = []

        for cat in categories:
            target  = cat.nominal_budget(usable)
            fill    = 100 * cat.allocated_weight / (target or 1)
            gt_info = gt.get(cat.name, {})
            give_v  = gt_info.get("give", 0.0)
            take_v  = gt_info.get("take", 0.0)
            net     = take_v - give_v
            net_str = (f"  ›give={give_v:.1f}w  take={take_v:.1f}w  "
                       f"net={'+'if net >= 0 else ''}{net:.1f}w") if gt_info else ""

            sep = (f"--- {cat.name}  "
                   f"önem={cat.importance:.2f}  "
                   f"oran=%{cat.allocation_ratio * 100:.0f}  "
                   f"hedef={target:.1f}w  "
                   f"kullanılan={cat.allocated_weight:.2f}w  "
                   f"doluluk=%{fill:.1f}"
                   + net_str)

            if HAS_TABULATE:
                rows.append([sep] + [""] * (len(self._HDR) - 1))
            else:
                lines.append(sep)

            for item in cat.items:
                d  = item.delta_from_baseline
                ds = f"+{d}" if d > 0 else (str(d) if d else "±0")
                row = [
                    f"  {item.name}",
                    str(item.type_count),
                    str(item.qty_per_type),
                    str(item.optimal_quantity),
                    f"{item.ideal_quantity:.1f}",
                    ds,
                    f"[{item.min_total}..{item.max_total}]",
                    str(item.preferred_min),
                    str(item.preferred_max),
                    str(item.deviation_positive) if item.deviation_positive else "–",
                    str(item.deviation_negative) if item.deviation_negative else "–",
                    f"{item.effective_weight:.3f}",
                    f"{item.total_weight:.2f}",
                    f"{item.steal_factor:.2f}",
                    f"{item.composite_importance:.2f}",
                    "★" if item.is_essential else "",
                ]
                if HAS_TABULATE:
                    rows.append(row)
                else:
                    lines.append("  ".join(str(c).ljust(w)
                                           for c, w in zip(row, self._COLW)))

            if HAS_TABULATE:
                rows.append([""] * len(self._HDR))
            else:
                lines.append("")

        if HAS_TABULATE:
            lines.append(tabulate(rows, headers=self._HDR, tablefmt="simple"))

        lines.append("═" * W)
        lines.append(
            "  TC=type_count  Q/TC=miktar_per_tip  Ideal=hedef_miktar  "
            "ΔBase=optimal−baseline  d+=PMax_üstü  d−=PMin_altı  "
            "SF=steal_factor  net=alınan−verilen_bütçe"
        )
        lines.append("═" * W)
        return "\n".join(lines)



#  JSON DIŞA AKTARICI


class JSONExporter:
    """Sonuçları optimization_result.json olarak kaydeder."""
    def export(
        self,
        categories:     list[Category],
        stash_capacity: float,
        buffer:         float,
        strategy:       str,
        out_path:       str | Path,
    ) -> None:
        usable    = stash_capacity - buffer
        all_items = [i for cat in categories for i in cat.items]
        used      = sum(i.total_weight for i in all_items)

        output = {
            "meta": {
                "strategy":          strategy,
                "stash_capacity":    stash_capacity,
                "buffer":            buffer,
                "usable_capacity":   usable,
                "total_weight_used": round(used, 4),
                "total_weight_free": round(usable - used, 4),
                "utilization_pct":   round(100 * used / usable, 2),
            },
            "categories": [],
        }

        for cat in categories:
            target = cat.nominal_budget(usable)
            cat_out: dict = {
                "category":            cat.name,
                "importance":          cat.importance,
                "allocation_ratio":    cat.allocation_ratio,
                "allocation_target":   round(target, 4),
                "allocated_weight":    round(cat.allocated_weight, 4),
                "allocation_fill_pct": round(
                    100 * cat.allocated_weight / (target or 1), 2
                ),
                "items": [],
            }

            for item in cat.items:
                cat_out["items"].append({
                    "name":                 item.name,
                    "type_count":           item.type_count,
                    "optimal_quantity":     item.optimal_quantity,
                    "ideal_quantity":       round(item.ideal_quantity, 3),
                    "baseline_quantity":    item.baseline_quantity,
                    "delta_from_baseline":  item.delta_from_baseline,
                    "min_total":            item.min_total,
                    "max_total":            item.max_total,
                    "preferred_min":        item.preferred_min,
                    "preferred_max":        item.preferred_max,
                    "deviation_positive":   item.deviation_positive,
                    "deviation_negative":   item.deviation_negative,
                    "qty_per_type":         item.qty_per_type,
                    "unit_weight":          item.unit_weight,
                    "effective_weight":     round(item.effective_weight, 4),
                    "total_weight":         round(item.total_weight, 4),
                    "is_essential":         item.is_essential,
                    "steal_factor":         item.steal_factor,
                    "composite_importance": round(item.composite_importance, 4),
                })

            output["categories"].append(cat_out)

        Path(out_path).write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[INFO] JSON kaydedildi → {out_path}")



#  ANA ORKESTRATÖRü


class StashOptimizer:
    """
    Kullanım:
        StashOptimizer("inventory.json").run()
        StashOptimizer("inventory.json", strategy="greedy").run()
    """

    STRATEGIES = ("scip", "greedy")

    def __init__(
        self,
        json_path: str | Path,
        strategy:  str = "scip",
        out_path:  str | Path | None = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.strategy  = strategy.lower()
        self.out_path  = (Path(out_path) if out_path
                          else self.json_path.parent / "optimization_result.json")

        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"Strateji {self.strategy!r} geçersiz. "
                             f"Seçenekler: {self.STRATEGIES}")

    def run(self) -> None:
        # Yükle
        stash_capacity, buffer, categories = InventoryLoader.load(self.json_path)
        usable    = stash_capacity - buffer
        all_items = [i for cat in categories for i in cat.items]

        print(f"[INFO] {len(categories)} kategori, {len(all_items)} item yüklendi.")
        print(f"[INFO] Kapasite={stash_capacity:.0f}  Buffer={buffer:.0f}  "
              f"Kullanılabilir={usable:.0f}")

        # Faz 1: İdeal hesapla
        print("[INFO] Faz 1 – İdeal hesaplama (cascade + global rescale) …")
        IdealCalculator(usable, categories).compute()

        # Faz 2: Optimize et
        use_scip = self.strategy == "scip" and HAS_SCIP
        if use_scip:
            print("[INFO] Faz 2 – SCIP MIQCP optimizasyonu …")
            opt   = SCIPOptimizer(all_items, usable, categories)
            label = "SCIP-v1.0"
        else:
            if self.strategy == "scip" and not HAS_SCIP:
                print("[WARN] SCIP bulunamadı, Greedy fallback kullanılıyor.",
                      file=sys.stderr)
            print("[INFO] Faz 2 – Greedy optimizasyon …")
            opt   = GreedyFallback(all_items, usable, categories)
            label = "GREEDY-v1.0"

        opt.optimize()
        give_take = opt.give_take_report()

        # Çıktılar
        report = ReportRenderer().render(
            categories, stash_capacity, buffer, label, give_take
        )
        print(report)

        JSONExporter().export(categories, stash_capacity, buffer, label, self.out_path)



if __name__ == "__main__":
    import argparse
    INVENTORY_PATH = Path(__file__).parent / "../data/inventory.json"
    # INVENTORY_PATH = Path(__file__).parent / "../data/inventory_test.json"
    p = argparse.ArgumentParser(description="Fallout 76 Stash Inventory Optimizer v1.0")
    p.add_argument(
        "--inventory",
        default=INVENTORY_PATH,
        help="inventory.json dosyasının yolu (varsayılan: ./inventory.json)",
    )
    p.add_argument(
        "--strategy",
        default="scip",
        choices=StashOptimizer.STRATEGIES,
        help="Optimizasyon stratejisi (varsayılan: scip)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Çıktı JSON yolu (varsayılan: inventory.json ile aynı klasör)",
    )
    args = p.parse_args()

    StashOptimizer(
        json_path = args.inventory,
        strategy  = args.strategy,
        out_path  = args.out,
    ).run()