"""
# Fallout 76 - Stash Inventory Optimizer v1.0.5
@Author: Burakhan Şamlı
@GitHub: https://github.com/blitzkrieg0000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d → %(message)s",
    datefmt="%d/%m/%Y - %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)],
)

try:
    from pyscipopt import SCIP_PARAMSETTING, Model, quicksum

    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False
    print("[WARN] pyscipopt bulunamadı - Greedy fallback kullanılacak.", file=sys.stderr)

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

_EPS = 1e-9

# IdealCalculator
IMPORTANCE_ALPHA: float = 1.6  # importance exponent (1=lineer, 2=quadratic, 3=cubic)
CASCADE_PASSES: int = 50  # cascade yeniden dağıtım geçişleri

# Kapasite doluluk bandı
# alt sınır: usable × bu oran ≥ Σw·q =>(0.95 -> %5 esneklik, her zaman feasible)
FILL_TOLERANCE: float = 0.95

# Objective ağırlıkları
ESSENTIAL_BONUS = 0.0  # TODO: Essential itemlar biraz daha önemli olsa, aynı derecede öneme sahip non-essential itemlar ile karışıklık olur mu? 0.0 -> 0.15
W_ITEM_DEV: float = 1.0  # sapma penalty temel çarpanı
W_FLOOR_ESS: float = 14.0  # essential altı zemin penalty (hard)
W_FLOOR_NON: float = 2.5  # non-essential altı zemin penalty (smooth)
W_EMPTY: float = 60.0  # boş kapasite penalty (her birim için)
W_CAT_GIVE: float = 5.0  # kategori bütçe verme penalty temel çarpanı
W_CAT_TAKE: float = 3.0  # kategori bütçe alma penalty temel çarpanı
# Σgive≈Σtake soft denge penalty (alınan verilene eşit)
W_POOL_BAL: float = 8.0

# Sınırlar
# essential item'ın mutlak min'i = min_total × <multiplier>
ESSENTIAL_HARD_FLOOR: float = 0.80
SURPLUS_LOG_BASE: float = 2.4  # surplus varsa max aşım faktörü
MAX_TAKE_FRACTION: float = 0.60  # bir kategorinin alabileceği max ek bütçe oranı

# Continuous relaxation eşikleri
# Geniş aralıklı + hafif itemlar (ammo vb.) -> continuous değişken, sonra yuvarla.
# B&B ağacını dramatik küçültür: ~135 integer -> ~50 integer değişken.
CONT_RANGE_THRESHOLD: int = 500  # max_total - min_total > bu ise aday
# effective_weight < bu ise continuous kullan
CONT_WEIGHT_THRESHOLD: float = 0.05


@dataclass
class InventoryItem:
    """Envanterdeki tek bir item grubu."""

    name: str
    category: str
    type_count: int  # aynı ağırlıktaki benzer ürün adedi
    unit_weight: float  # bir birimin ağırlığı
    weight_modifier: float  # ağırlık çarpanı (default 1.0)

    min_unit_quantity: int  # bir tip için minimum miktar
    max_unit_quantity: int  # bir tip için maksimum miktar

    category_importance: float  # kategorinin önemi [0,1]
    item_importance: float  # itemın kendi önemi [0,1]
    is_essential: bool  # essential ise zemin koruması daha güçlü
    steal_factor: float  # 0→az, 1→çok - başkalarından ne kadar yer alır

    # IdealCalculator tarafından doldurulur
    ideal_quantity: float = 0.0  # hedef miktar (float, kesirsiz hesap)
    baseline_quantity: int = 0  # ideal'in yuvarlanmış hali

    # Solver tarafından doldurulur
    optimal_quantity: int = 0
    preferred_min: int = 0
    preferred_max: int = 0
    deviation_positive: int = 0  # d+  -> preferred_max üstünde
    deviation_negative: int = 0  # d-  -> preferred_min altında

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
            base = min(1.0, base + ESSENTIAL_BONUS)
        return round(min(1.0, max(0.01, base)), 6)

    @property
    def delta_from_baseline(self) -> int:
        return self.optimal_quantity - self.baseline_quantity


@dataclass
class Category:
    """Envanter kategorisi."""

    name: str
    importance: float
    allocation_ratio: float  # usable'ın bu oranı bu kategoriye ayrılır
    max_capacity: Optional[float] = None  # isteğe bağlı hard üst sınır
    items: list[InventoryItem] = field(default_factory=list)

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


class InventoryLoader:
    """inventory.json'u okuyup Category + InventoryItem listesi döner."""

    @staticmethod
    def load(path: str | Path) -> tuple[float, float, list[Category]]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        capacity: float = float(raw["stash_capacity"])
        buffer: float = float(raw["buffer"])
        usable = capacity - buffer

        if usable <= 0:
            raise ValueError("buffer ≥ stash_capacity! Kullanılabilir alan yok.")

        categories: list[Category] = []
        for cd in raw["categories"]:
            cat_imp = float(cd["category_importance"])
            alloc_rat = float(cd["allocation_ratio"])

            items: list[InventoryItem] = []
            for ri in cd["items"]:
                item = InventoryItem(
                    name=str(ri["name"]),
                    category=str(cd["category"]),
                    type_count=int(ri["type_count"]),
                    unit_weight=float(ri["unit_weight"]),
                    weight_modifier=float(ri.get("weight_modifier", 1.0)),
                    min_unit_quantity=int(ri["min_unit_quantity"]),
                    max_unit_quantity=int(ri["max_unit_quantity"]),
                    category_importance=cat_imp,
                    item_importance=float(ri["item_importance"]),
                    is_essential=bool(ri.get("is_essential", False)),
                    steal_factor=float(ri.get("steal_factor", 0.5)),
                )
                items.append(item)

            categories.append(
                Category(
                    name=str(cd["category"]),
                    importance=cat_imp,
                    allocation_ratio=alloc_rat,
                    max_capacity=cd.get("max_capacity"),
                    items=items,
                )
            )

        # allocation_ratio toplamı 1'den büyükse uyar ama hata verme
        total_ratio = sum(c.allocation_ratio for c in categories)
        if total_ratio > 1.0 + _EPS:
            print(f"[WARN] allocation_ratio toplamı {total_ratio:.3f} > 1.0 - kategoriler birbirleriyle rekabet eder.", file=sys.stderr)

        return (capacity, buffer, categories)


class IdealCalculator:
    """
    İki fazlı ideal miktar hesabı:

    Faz A - _cascade():
        Her kategorinin nominal bütçesi içinde importance^alpha ağırlıklı
        orantılı başlangıç dağılımı yapılır. Havuz yeniden dağıtımı
        CASCADE_PASSES turunda yakınsayana kadar tekrarlanır.
        Taşan havuz self.global_pool'a aktarılır.

    Faz B - _global_rescale():
        self.global_pool, tüm itemlara önem derecelerine göre dağıtılır.
        Yakınsama sonrası kalan küçük artık hard-clamp ile temizlenir.
    """

    def __init__(self, usable: float, categories: list[Category]) -> None:
        self.usable = usable
        self.categories = categories
        self.local_penalty = False
        self.global_pool = 0.0

    def compute(self) -> None:
        for cat in self.categories:
            self._cascade(cat)
        logging.debug(f"[after-cascade] Final Global Pool(w): {self.global_pool:g}")
        self._global_rescale()

    #! FAZ A - Kategori içi cascade
    def _cascade(self, cat: Category) -> float:
        """
        Kategorinin nominal bütçesini importance orantılı dağıtır.
        Taşan havuz self.global_pool'a yazılır.
        Dönüş: residual (bütçe - kullanılan ağırlık).
        """
        items = cat.items
        if not items:
            return 0.0

        cat_budget = cat.nominal_budget(self.usable)
        scores = [i.composite_importance**IMPORTANCE_ALPHA for i in items]
        score_sum = sum(scores) or 1.0

        # Başlangıç dağılımı: importance orantılı (miktar cinsinden)
        alloc: list[float] = [(s / score_sum) * cat_budget / (i.effective_weight or _EPS) for s, i in zip(scores, items)]

        # Havuz dağıtımı (cascade)
        carry = self._distribute(items, scores, alloc, seed_pool=self.global_pool)
        self.global_pool = carry  # dağıtılamayan -> global havuza

        # Eşit önem + ağırlık + sınır -> tam ortalama garantisi
        if self._all_equal(items) and len(items) > 1:
            avg = sum(alloc) / len(alloc)
            alloc = [avg] * len(alloc)

        # item'ları güncelle
        used = self._commit_alloc(items, alloc)

        # local_penalty: used > budget ise orantılı küçült
        if self.local_penalty and used > cat_budget + _EPS:
            used = self._apply_local_penalty(items, cat_budget, used)

        residual = cat_budget - used
        _approx = lambda x: "0" if abs(x) < _EPS else f"{x:g}"
        logging.debug(
            f"[cascade] category='{cat.name}', budget(w)={cat_budget:g}, "
            f"used(w)={used:g}, residual={_approx(residual)}, "
            f"global_pool(w)={_approx(self.global_pool)}"
        )
        return residual

    #! FAZ B - Importance bazlı Global pool dağıtımı
    def _global_rescale(self) -> None:
        """
        self.global_pool'u tüm itemlara importance orantılı dağıtır.
        Yakınsama sonrası kalan artık için hard-clamp fallback uygulanır.
        """
        all_items = [i for cat in self.categories for i in cat.items]
        if not all_items:
            return

        if abs(self.global_pool) < _EPS:
            self._commit_alloc(all_items, [i.ideal_quantity for i in all_items])
            return

        scores = [i.composite_importance**IMPORTANCE_ALPHA for i in all_items]
        alloc = [i.ideal_quantity for i in all_items]  # mevcut idealden başla

        carry = self._distribute(all_items, scores, alloc, seed_pool=self.global_pool)

        # Yakınsama sonrası artık -> usable'ı aşmayan hard-clamp scale
        if abs(carry) > _EPS:
            logging.warning(f"[global_rescale] yakınsama artığı carry(w)={carry:g} - hard-clamp uygulanıyor.")
            total_w = sum(alloc[i] * all_items[i].effective_weight for i in range(len(alloc)))
            target_w = min(self.usable, total_w + carry)
            scale = target_w / total_w if total_w > _EPS else 1.0
            alloc = [max(float(all_items[i].min_total), min(float(all_items[i].max_total), alloc[i] * scale)) for i in range(len(alloc))]

        used = self._commit_alloc(all_items, alloc)
        logging.debug(
            f"[global_rescale] items={len(all_items)}, "
            f"global_pool(w)={self.global_pool:g}, "
            f"final_used(w)={used:g}, usable(w)={self.usable:g}, "
            f"residual(w)={self.usable - used:g}"
        )

    # Havuz dağıtımı
    @staticmethod
    def _distribute(
        items: list[InventoryItem], scores: list[float], alloc: list[float], seed_pool: float = 0.0  # in-place güncellenir
    ) -> float:
        """
        Verilen alloc listesini min/max sınırlarına clamp'leyerek
        seed_pool'u CASCADE_PASSES turunda importance orantılı dağıtır.

        pool > 0 → max'a çarpmamış adaylara importance orantılı ver.
        pool < 0 → min'in üstündeki adaylardan importance ters orantılı kes.

        Dönüş: dağıtılamayan carry (yakınsama artığı).
        """
        carry = seed_pool

        for _ in range(CASCADE_PASSES):
            pool = carry
            carry = 0.0

            # Sınır denetimleri + clamp
            for idx, item in enumerate(items):
                qty = alloc[idx]
                if qty > item.max_total:
                    pool += (qty - item.max_total) * item.effective_weight
                    alloc[idx] = float(item.max_total)
                elif qty < item.min_total:
                    pool -= (item.min_total - qty) * item.effective_weight
                    alloc[idx] = float(item.min_total)

            if abs(pool) < _EPS:
                break

            # Aday seçimi + ağırlık hesabı
            if pool > 0:
                cands = [i for i in range(len(items)) if alloc[i] < items[i].max_total]
                cweights = [scores[i] for i in cands]
            else:
                cands = [i for i in range(len(items)) if alloc[i] > items[i].min_total]
                cweights = [1.0 / (scores[i] + _EPS) for i in cands]

            if not cands:
                carry = pool  # tüm adaylar sınırda, taşan miktar geri döner
                break

            w_sum = sum(cweights) or 1.0
            absorbed = 0.0
            for k, j in enumerate(cands):
                delta = (cweights[k] / w_sum) * pool / (items[j].effective_weight or _EPS)
                before = alloc[j]
                alloc[j] = max(float(items[j].min_total), min(float(items[j].max_total), alloc[j] + delta))
                absorbed += (alloc[j] - before) * items[j].effective_weight

            carry = pool - absorbed
            if abs(carry) < _EPS:
                break

        return carry

    def _commit_alloc(self, items: list[InventoryItem], alloc: list[float]) -> float:
        """
        alloc listesi ile item'lar güncellenir (ideal, baseline, optimal, preferred).
        Dönüş: toplam kullanılan ağırlık.
        """
        used = 0.0
        for item, qty in zip(items, alloc):
            qty = max(float(item.min_total), min(float(item.max_total), qty))
            item.ideal_quantity = qty
            item.baseline_quantity = int(round(qty))
            item.optimal_quantity = item.baseline_quantity
            used += qty * item.effective_weight
            self._set_preferred(item)
        return used

    def _apply_local_penalty(self, items: list[InventoryItem], cat_budget: float, used: float) -> float:
        """
        used > budget durumunda orantılı küçültme (local_penalty=True).
        Diğer kategoriler bu taşmadan ceza almasın diye budget'a çeker.
        Dönüş: düzeltilmiş used (= cat_budget).
        """
        scale = cat_budget / used
        logging.warning(f"[local_penalty] min_weight ({used:.2f}) > budget ({cat_budget:.2f}), " f"scale={scale:.4f} uygulanıyor.")
        for item in items:
            scaled = item.ideal_quantity * scale
            item.ideal_quantity = scaled
            item.baseline_quantity = int(round(scaled))
            item.optimal_quantity = item.baseline_quantity
            self._set_preferred(item)
        return cat_budget  # used artık budget'a eşit

    @staticmethod
    def _all_equal(items: list[InventoryItem]) -> bool:
        """Tüm itemlar aynı önem/ağırlık/sınıra sahipse True."""
        return (
            len({i.composite_importance for i in items}) == 1
            and len({i.effective_weight for i in items}) == 1
            and len({i.min_total for i in items}) == 1
            and len({i.max_total for i in items}) == 1
        )

    @staticmethod
    def _set_preferred(item: InventoryItem) -> None:
        """
        preferred_min/max: ideal ±%30 penceresi (sadece raporlama amaçlı).
        - essential → pmin, hard floor ile korunur (min_total × ESSENTIAL_HARD_FLOOR).
        - non-essential → pmin ≥ 0.
        - pmax her zaman max_total ile kırpılır.
        """
        flex = 0.30
        pmin = item.ideal_quantity * (1.0 - flex)
        pmax = item.ideal_quantity * (1.0 + flex)

        if item.is_essential:
            hard_floor = float(item.min_total) * ESSENTIAL_HARD_FLOOR
            pmin = max(pmin, hard_floor)
        else:
            pmin = max(0.0, pmin)

        item.preferred_min = int(math.floor(pmin))
        item.preferred_max = int(math.ceil(min(float(item.max_total), pmax)))


class SCIPOptimizer:
    """
    #! MIQCP (Mixed-Integer Quadratically-Constrained) optimizasyonu.

    PySCIPOpt kısıtı
    Objective fonksiyonu LINEAR olmalı.
    Kısıtlar QUADRATIC olabilir (MIQCP).

    Değişkenler
    ---
    q_i   ∈ ℤ           : item miktarları
    give_c, take_c ≥ 0  : kategori bütçe akış değişkenleri (soft)
    slack ≥ 0            : boş kapasite (penalty)
    below_i ≥ 0          : min_total altına iniş miktarı (penalty)
    d_i (serbest)        : idealden sapma


    Hard kısıtlar
    cap_hi  : Σ w·q ≤ usable
    cap_lo  : Σ w·q ≥ usable × FILL_TOLERANCE
    q_i ≥ hard_lb_i      (essential -> ESSENTIAL_HARD_FLOOR × min_total)
    q_i ≤ hard_ub_i      (surplus varsa max aşımına izin)

    Soft amaç (minimize)
    1) Normalize sapma  : norm_w_i × d_i²
    norm_w_i = (1/(imp·sf)) / Σ(1/(imp·sf)) × N
    -> tüm itemlardan orantılı pay; tek itemdan çalınma yok

    2) Logaritmik zemin : α × (below² + below)
    -> below→0 iken bile linear gradient var; 0'a yapışma yok

    3) Boş alan         : W_EMPTY × slack

    4) Kategori give/take (asimetrik):
    give cezası = W_CAT_GIVE × (1−imp)² × give²  (önemsiz -> ucuz verir)
    take cezası = W_CAT_TAKE × (1−imp)  × take²  (önemli -> ucuz alır)
    pool balance= W_POOL_BAL × (Σgive - Σtake)²  (soft denge)
    """

    def __init__(self, items: list[InventoryItem], usable: float, categories: list[Category]) -> None:
        self.items = items
        self.usable = usable
        self.categories = categories

        # Kategori -> item eşlemesi
        self._cat_items: dict[str, list[InventoryItem]] = {}
        for item in items:
            self._cat_items.setdefault(item.category, []).append(item)

        # Surplus (tüm max_total ağırlıklarının usable'dan küçük olduğu fazlalık)
        max_w = sum(i.effective_weight * i.max_total for i in items)
        self._surplus = max(0.0, usable - max_w)

        # Normalize edilmiş sapma ağırlıkları (smooth spreading)
        # önemli item -> yüksek ceza -> zor kesilir; yüksek sf -> düşük ceza -> kolay kesilir

        #! Asimetrik deviation weights
        # Yokluk (below ideal): imp × sf -> yüksek sf -> dik ceza -> zor kesilir
        # Varlık  (above ideal): imp / sf -> yüksek sf -> düşük ceza -> rahat genişler
        N = len(items)
        raw_neg = [W_ITEM_DEV * i.composite_importance * (i.steal_factor + _EPS) for i in items]  # Steal Factor ile doğru orantılı
        raw_pos = [W_ITEM_DEV * i.composite_importance / (i.steal_factor + _EPS) for i in items]  # Steal Factor ile ters orantılı
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
        frac = math.log(1.0 + SURPLUS_LOG_BASE * self._surplus / (self.usable + _EPS))
        bonus = frac * item.composite_importance
        return int(math.ceil(item.max_total * (1.0 + bonus)))

    # Yardımcı: integer mi continuous mi?
    def _use_continuous(self, item: InventoryItem) -> bool:
        """
        Ammo gibi geniş aralıklı ama birim ağırlığı küçük itemlar için
        continuous (C) değişken kullan, sonunda integer'a yuvarla.

        Kriter: (max_total - min_total) > CONT_RANGE_THRESHOLD  VE effective_weight < CONT_WEIGHT_THRESHOLD
        Bu itemlarda integrality constraint B&B ağacını gereksiz büyütür
        ama çözüme katkısı birim ağırlık başına çok küçük olduğundan
        yuvarlama hatası ihmal edilebilir.
        """
        return (item.max_total - item.min_total) > CONT_RANGE_THRESHOLD and item.effective_weight < CONT_WEIGHT_THRESHOLD

    # Ana optimizasyon fonksiyonu
    def optimize(self) -> None:
        if not HAS_SCIP:
            raise RuntimeError("pyscipopt kurulu değil.")

        m = Model("StashOptimizer_v1")
        m.hideOutput()
        m.setPresolve(SCIP_PARAMSETTING.AGGRESSIVE)
        m.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)

        #! item miktar değişkenleri
        # Geniş aralıklı + hafif itemlar -> continuous (C), diğerleri -> integer (I)
        # Continuous olanlar sonradan integer'a yuvarlanır; B&B ağacı dramatik ölçüde küçülür.
        q: dict[int, object] = {}
        for item in self.items:
            iid = id(item)
            vtype = "C" if self._use_continuous(item) else "I"
            q[iid] = m.addVar(vtype=vtype, lb=float(self._hard_lb(item)), ub=float(self._hard_ub(item)), name=f"q_{item.name[:18]}_{iid}")

        #! Kapasite band kısıtı (HARD)
        # itemlar %95 ile %100 arasında olmalıdır.
        total_w = quicksum(i.effective_weight * q[id(i)] for i in self.items)
        m.addCons(total_w <= self.usable, name="cap_hi")
        m.addCons(total_w >= self.usable * FILL_TOLERANCE, name="cap_lo")

        #! Boş alan ceza sınırı
        # Boş alan ne kadar fazla ise o kadar ceza verilir.
        slack = m.addVar(vtype="C", lb=0.0, ub=self.usable * (1.0 - FILL_TOLERANCE) + 1.0, name="slack")
        m.addCons(slack >= self.usable - total_w, name="slack_def")

        #! Kategori give/take (SOFT, linear approx)
        # Kategoriler arası alan alıp/verme kısıtları. Alınan alan verilene eşit olmalıdır.
        give: dict[str, object] = {}
        take: dict[str, object] = {}
        for cat in self.categories:
            cid = cat.name
            items_c = self._cat_items.get(cid, [])
            if not items_c:
                continue
            T_c = cat.nominal_budget(self.usable)
            imp = cat.importance

            give[cid] = m.addVar(vtype="C", lb=0.0, ub=T_c, name=f"give_{cid}")
            take[cid] = m.addVar(vtype="C", lb=0.0, ub=MAX_TAKE_FRACTION * T_c, name=f"take_{cid}")

        # Pool balance: Σgive == Σtake (hard equality, linear)
        cids = list(give.keys())
        if len(cids) > 1:
            m.addCons(quicksum(give[c] for c in cids) == quicksum(take[c] for c in cids), name="pool_balance")

        #! Asimetrik sapma penalty (QP objective via epigraph)
        # d_neg_i = max(0, ideal_i - q_i)   -> yokluk cezası  (w_neg: imp × sf)
        # d_pos_i = max(0, q_i - ideal_i)   -> varlık cezası  (w_pos: imp / sf)
        #
        # Her QP terimi: epigraph değişkeni haline getirilir: t ile  t ≥ w·d²  (quadratic constraint)
        t_dev_neg: dict[int, object] = {}
        t_dev_pos: dict[int, object] = {}

        for idx, item in enumerate(self.items):
            iid = id(item)
            w_n = self._w_neg[idx]
            w_p = self._w_pos[idx]

            d_neg = m.addVar(vtype="C", lb=0.0, name=f"dneg_{iid}")
            d_pos = m.addVar(vtype="C", lb=0.0, name=f"dpos_{iid}")
            m.addCons(d_neg >= item.ideal_quantity - q[iid], name=f"dneg_{iid}")
            m.addCons(d_pos >= q[iid] - item.ideal_quantity, name=f"dpos_{iid}")

            # Steal Factor düşükse item ideal altına düşebilir ancak yüksemesi daha fazla ceza puanı ekler.        # w_n=0.1, w_p=10
            # Steal Factor yüksekse, item ideal altına düşmesi zorlaşır ancak yükselmesi daha az ceza puanı ekler. # w_n=0.9, w_p=1.1
            tn = m.addVar(vtype="C", lb=0.0, name=f"tn_{iid}")
            tp = m.addVar(vtype="C", lb=0.0, name=f"tp_{iid}")
            m.addCons(tn >= w_n * d_neg * d_neg, name=f"tn_qp_{iid}") 
            m.addCons(tp >= w_p * d_pos * d_pos, name=f"tp_qp_{iid}")  
            t_dev_neg[iid] = tn
            t_dev_pos[iid] = tp

        #! Logaritmik zemin penalty
        # α×(below² + below): below=0'da bile α gradient -> 0'a yapışmaz
        t_floor: dict[int, object] = {}
        for item in self.items:
            if item.min_total <= 0:
                continue
            iid = id(item)
            alpha = W_FLOOR_ESS if item.is_essential else W_FLOOR_NON

            below = m.addVar(vtype="C", lb=0.0, name=f"bel_{iid}")
            m.addCons(below >= item.min_total - q[iid], name=f"bel_{iid}")

            tf = m.addVar(vtype="C", lb=0.0, name=f"tf_{iid}")
            m.addCons(tf >= alpha * (below * below + below), name=f"tf_qp_{iid}")
            t_floor[iid] = tf

        #! Lineer give/take ceza terimi
        # Quadratic give/take yerine lineer ceza: (1-imp)² × give + (1-imp) × take
        # Önemsiz itemlar kolay yer verir), QP constraint 0'a düşer.
        give_lin_cost = quicksum(
            W_CAT_GIVE * (1.0 - self.categories[i].importance) ** 2 * give[cid]
            for i, cat in enumerate(self.categories)
            for cid in [cat.name]
            if cid in give
        )

        take_lin_cost = quicksum(
            W_CAT_TAKE * (1.0 - self.categories[i].importance) * take[cid]
            for i, cat in enumerate(self.categories)
            for cid in [cat.name]
            if cid in take
        )

        #! Objective Function
        obj = (
            quicksum(t_dev_neg[iid] for iid in t_dev_neg)
            + quicksum(t_dev_pos[iid] for iid in t_dev_pos)
            + quicksum(t_floor[iid] for iid in t_floor)
            + W_EMPTY * slack
            + give_lin_cost
            + take_lin_cost
        )
        m.setObjective(obj, "minimize")

        #! Solve
        m.optimize()

        status = m.getStatus()
        if status not in ("optimal", "bestsolfound"):
            print(f"[WARN] SCIP status={status!r} - IdealCalculator baseline kullanılıyor.", file=sys.stderr)
            self._finalise()
            return

        # Sonuçları yaz - continuous olanlar integer'a yuvarla
        for item in self.items:
            val = m.getVal(q[id(item)])
            item.optimal_quantity = max(self._hard_lb(item), min(self._hard_ub(item), int(round(val))))

        # Give/take değerlerini kaydet (raporlama için)
        for c in give:
            self._give_vals[c] = m.getVal(give[c])
            self._take_vals[c] = m.getVal(take[c])

        self._finalise()

    def _finalise(self) -> None:
        for item in self.items:
            item.deviation_positive = max(0, item.optimal_quantity - item.preferred_max)
            item.deviation_negative = max(0, item.preferred_min - item.optimal_quantity)

    def give_take_report(self) -> dict[str, dict[str, float]]:
        """Solver sonrası kategori bütçe akış raporu."""
        return {c: {"give": self._give_vals.get(c, 0.0), "take": self._take_vals.get(c, 0.0)} for c in self._give_vals}


class GreedyFallback:
    """
    Basit greedy algoritma.
    1. Baseline'dan başla.
    2. Kapasite aşılıyorsa -> önemsizlerden kes.
    3. Kapasite altındaysa -> önemlileri şişir.
    """

    def __init__(self, items: list[InventoryItem], usable: float, categories: list[Category]) -> None:
        self.items = items
        self.usable = usable
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
        """Kapasiteyi aşan ağırlığı önemsizlerden arındır."""
        for _ in range(1000):
            over = self._total_w() - self.usable
            if over <= 0:
                break

            candidates = []
            for item in self.items:
                floor = int(item.min_total * ESSENTIAL_HARD_FLOOR) if item.is_essential else 0
                room = item.optimal_quantity - floor
                if room <= 0:
                    continue
                # Önemsiz itemdan daha çok kes (steal_weight yüksekse ucuz)
                sw = (
                    ((1.0 - item.category_importance) + (1.0 - item.composite_importance) + (1.0 - item.steal_factor))
                    * room
                    * item.effective_weight
                )  # yüksek sf -> zor kurban -> az kesilir
                candidates.append((item, sw, floor))

            if not candidates:
                break

            total_sw = sum(sw for _, sw, _ in candidates) or 1.0
            for item, sw, floor in candidates:
                cut_w = over * (sw / total_sw)
                cut_u = min(item.optimal_quantity - floor, max(0, int(math.ceil(cut_w / (item.effective_weight or 1)))))
                item.optimal_quantity -= cut_u

    def _expand(self) -> None:
        """Boş alanı önemlileri şişirerek doldur."""
        max_w = sum(i.effective_weight * i.max_total for i in self.items)
        glob_surp = max(0.0, self.usable - max_w)

        def ub(item: InventoryItem) -> int:
            if glob_surp <= _EPS:
                return item.max_total
            log_s = math.log(1.0 + SURPLUS_LOG_BASE * glob_surp / (self.usable + _EPS))
            return item.max_total + int(log_s * item.composite_importance * item.max_total)

        # Sıralama: category_importance, sonra composite_importance × steal_factor
        # -> yüksek sf item artakalan alanı önce kapıyor (agresif genişleme)
        ranked = sorted(self.items, key=lambda i: (i.category_importance, i.composite_importance * i.steal_factor), reverse=True)
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
            item.deviation_negative = max(0, item.preferred_min - item.optimal_quantity)

    def give_take_report(self) -> dict[str, dict[str, float]]:
        return {}  # Greedy'de give/take yok


class ReportRenderer:
    """Sonuçları konsola tablo olarak basar."""

    _HDR = ["Name", "TC", "Q/TC", "TotalQ", "Ideal", "ΔBase", "Range", "PMin", "PMax", "d+", "d-", "UnitW", "TotalW", "SF", "Imp", "Ess"]
    _COLW = [26, 4, 6, 8, 8, 7, 15, 6, 6, 5, 5, 7, 8, 5, 5, 4]

    def render(
        self,
        categories: list[Category],
        stash_capacity: float,
        buffer: float,
        strategy: str,
        give_take: dict[str, dict[str, float]] | None = None,
    ) -> str:
        usable = stash_capacity - buffer
        all_items = [i for cat in categories for i in cat.items]
        used = sum(i.total_weight for i in all_items)
        gt = give_take or {}

        W = 160
        lines: list[str] = []
        lines.append("═" * W)
        lines.append(" FALLOUT 76 - STASH OPTİMİZER v1.0 ".center(W, "═"))
        lines.append("═" * W)
        lines.append(
            f"  Strateji={strategy}  Kapasite={stash_capacity:.0f}  Buffer={buffer:.0f}  Kullanılabilir={usable:.0f}  Kullanılan={used:.2f}  Boş={usable - used:.2f}  ({100 * used / usable:.1f}%)"
        )
        lines.append("═" * W)

        rows: list[list[str]] = []

        for cat in categories:
            target = cat.nominal_budget(usable)
            fill = 100 * cat.allocated_weight / (target or 1)
            gt_info = gt.get(cat.name, {})
            give_v = gt_info.get("give", 0.0)
            take_v = gt_info.get("take", 0.0)
            net = take_v - give_v
            net_str = (f"  ›give={give_v:.1f}w  take={take_v:.1f}w  net={'+' if net >= 0 else ''}{net:.1f}w") if gt_info else ""

            sep = (
                f"--- {cat.name}  önem={cat.importance:.2f}  oran=%{cat.allocation_ratio * 100:.0f}  hedef={target:.1f}w  kullanılan={cat.allocated_weight:.2f}w  doluluk=%{fill:.1f}"
                + net_str
            )

            if HAS_TABULATE:
                rows.append([sep] + [""] * (len(self._HDR) - 1))
            else:
                lines.append(sep)

            for item in cat.items:
                d = item.delta_from_baseline
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
                    (str(item.deviation_positive) if item.deviation_positive else "-"),
                    (str(item.deviation_negative) if item.deviation_negative else "-"),
                    f"{item.effective_weight:.3f}",
                    f"{item.total_weight:.2f}",
                    f"{item.steal_factor:.2f}",
                    f"{item.composite_importance:.2f}",
                    ("★" if item.is_essential else ""),
                ]
                if HAS_TABULATE:
                    rows.append(row)
                else:
                    lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, self._COLW)))

            if HAS_TABULATE:
                rows.append([""] * len(self._HDR))
            else:
                lines.append("")

        if HAS_TABULATE:
            lines.append(tabulate(rows, headers=self._HDR, tablefmt="simple"))

        lines.append("═" * W)
        lines.append(
            "  TC=type_count  Q/TC=miktar_per_tip  Ideal=hedef_miktar  ΔBase=optimal−baseline  d+=PMax_üstü  d−=PMin_altı  SF=steal_factor  net=alınan−verilen_bütçe"
        )
        lines.append("═" * W)
        return "\n".join(lines)


class JSONExporter:
    """Sonuçları optimization_result.json olarak kaydeder."""

    def export(self, categories: list[Category], stash_capacity: float, buffer: float, strategy: str, out_path: str | Path) -> None:
        usable = stash_capacity - buffer
        all_items = [i for cat in categories for i in cat.items]
        used = sum(i.total_weight for i in all_items)

        output = {
            "meta": {
                "strategy": strategy,
                "stash_capacity": stash_capacity,
                "buffer": buffer,
                "usable_capacity": usable,
                "total_weight_used": round(used, 4),
                "total_weight_free": round(usable - used, 4),
                "utilization_pct": round(100 * used / usable, 2),
            },
            "categories": [],
        }

        for cat in categories:
            target = cat.nominal_budget(usable)
            cat_out: dict = {
                "category": cat.name,
                "importance": cat.importance,
                "allocation_ratio": cat.allocation_ratio,
                "allocation_target": round(target, 4),
                "allocated_weight": round(cat.allocated_weight, 4),
                "allocation_fill_pct": round(100 * cat.allocated_weight / (target or 1), 2),
                "items": [],
            }

            for item in cat.items:
                cat_out["items"].append(
                    {
                        "name": item.name,
                        "type_count": item.type_count,
                        "optimal_quantity": item.optimal_quantity,
                        "ideal_quantity": round(item.ideal_quantity, 3),
                        "baseline_quantity": item.baseline_quantity,
                        "delta_from_baseline": item.delta_from_baseline,
                        "min_total": item.min_total,
                        "max_total": item.max_total,
                        "preferred_min": item.preferred_min,
                        "preferred_max": item.preferred_max,
                        "deviation_positive": item.deviation_positive,
                        "deviation_negative": item.deviation_negative,
                        "qty_per_type": item.qty_per_type,
                        "unit_weight": item.unit_weight,
                        "effective_weight": round(item.effective_weight, 4),
                        "total_weight": round(item.total_weight, 4),
                        "is_essential": item.is_essential,
                        "steal_factor": item.steal_factor,
                        "composite_importance": round(item.composite_importance, 4),
                    }
                )

            output["categories"].append(cat_out)

        Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] JSON kaydedildi -> {out_path}")


class StashOptimizer:
    STRATEGIES = ("scip", "greedy")

    def __init__(self, json_path: str | Path, strategy: str = "scip", out_path: str | Path | None = None) -> None:
        self.json_path = Path(json_path)
        self.strategy = strategy.lower()
        self.out_path = Path(out_path) if out_path else self.json_path.parent / "optimization_result.json"

        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"Strateji {self.strategy!r} geçersiz. Seçenekler: {self.STRATEGIES}")

    def run(self) -> None:
        (stash_capacity, buffer, categories) = InventoryLoader.load(self.json_path)
        usable = stash_capacity - buffer
        all_items = [i for cat in categories for i in cat.items]

        print(f"[INFO] {len(categories)} kategori, {len(all_items)} item yüklendi.")
        print(f"[INFO] Kapasite={stash_capacity:.0f}  Buffer={buffer:.0f}  Kullanılabilir={usable:.0f}")

        # Faz 1: İdeal hesapla
        print("[INFO] Faz 1 - İdeal hesaplama (cascade + global rescale) …")
        IdealCalculator(usable, categories).compute()

        # Faz 2: Optimize et
        use_scip = self.strategy == "scip" and HAS_SCIP
        if use_scip:
            print("[INFO] Faz 2 - SCIP MIQCP optimizasyonu …")
            opt = SCIPOptimizer(all_items, usable, categories)
            label = "SCIP-v1.0.5"
        else:
            if self.strategy == "scip" and not HAS_SCIP:
                print("[WARN] SCIP bulunamadı, Greedy fallback kullanılıyor.", file=sys.stderr)
            print("[INFO] Faz 2 - Greedy optimizasyon …")
            opt = GreedyFallback(all_items, usable, categories)
            label = "GREEDY-v1.0.5"

        opt.optimize()
        give_take = opt.give_take_report()

        report = ReportRenderer().render(categories, stash_capacity, buffer, label, give_take)
        print(report)

        JSONExporter().export(categories, stash_capacity, buffer, label, self.out_path)


if __name__ == "__main__":
    INVENTORY_PATH = Path(__file__).parent / "../data/inventory.json"
    INVENTORY_PATH = Path(__file__).parent / "../data/inventory-test.json"
    INVENTORY_PATH = Path(__file__).parent / "../data/inventory-custom.json"
    p = argparse.ArgumentParser(description="Fallout 76 Stash Inventory Optimizer v1.0")
    p.add_argument("--inventory", default=INVENTORY_PATH, help="inventory.json dosyasının yolu (varsayılan: ./inventory.json)")
    p.add_argument("--strategy", default="scip", choices=StashOptimizer.STRATEGIES, help="Optimizasyon stratejisi (varsayılan: scip)")
    p.add_argument("--out", default=None, help="Çıktı JSON yolu (varsayılan: inventory.json ile aynı klasör)")
    args = p.parse_args()

    StashOptimizer(json_path=args.inventory, strategy=args.strategy, out_path=args.out).run()
