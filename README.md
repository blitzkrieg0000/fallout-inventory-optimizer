# Fallout 76 - Stash Inventory Optimizer

A single-file Python optimizer that reads your inventory configuration from a JSON file and computes the optimal quantity for every item in your stash — respecting weight limits, importance priorities, min/max constraints, and cross-category budget dynamics.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [inventory.json Schema](#inventoryjson-schema)
- [Output](#output)
- [Architecture](#architecture)
- [Optimization Model](#optimization-model)
  - [Phase 1 - IdealCalculator](#phase-1--idealcalculator)
  - [Phase 2 - SCIP Optimizer (MIQCP)](#phase-2--scip-optimizer-miqcp)
  - [Asymmetric Deviation Weights](#asymmetric-deviation-weights)
  - [Logarithmic Floor Protection](#logarithmic-floor-protection)
  - [Category Give / Take](#category-give--take)
  - [Surplus Relaxation](#surplus-relaxation)
  - [Greedy Fallback](#greedy-fallback)
- [Tunable Constants](#tunable-constants)
- [What is SCIP?](#what-is-scip)
- [What is PySCIPOpt?](#what-is-pyscipopt)

---

## Overview

Your Fallout 76 stash has a hard weight cap. You want to keep hundreds of item types — junk, aid, ammo, weapons — each with different importances, minimum safe quantities, and maximum useful quantities. Deciding how much of each item to keep by hand is tedious and suboptimal.

This tool frames the problem as a **Mixed-Integer Quadratically-Constrained Program (MIQCP)** and solves it in a single pass using the SCIP solver, producing a fully filled stash where:

- important items are protected from cuts
- aggressive items (`steal_factor`) expand first when space opens up
- no item collapses to zero unexpectedly
- the total weight stays within your configured capacity

---

## Features

| Feature | Description |
|---|---|
| **Min / Max enforcement** | Each item stays within its configured quantity range |
| **Essential item protection** | Hard floor at `ESSENTIAL_HARD_FLOOR × min_total`; steep penalty below it |
| **Importance-proportional allocation** | Items receive space proportional to `composite_importance^α` |
| **Smooth spreading** | When space is scarce, cuts are distributed across all items — not stolen from a single one |
| **Asymmetric steal_factor** | High `steal_factor` = hard to cut (scarcity) AND first to expand (surplus) |
| **Logarithmic floor** | Items approaching zero face accelerating penalties — they never snap to 0 |
| **Cross-category budget flow** | Categories can give/take budget from each other, guided by their importance |
| **Surplus relaxation** | If all items are at max and space remains, important items may exceed `max_total` |
| **Capacity band constraint** | Total weight is guaranteed to fill ≥ 95% of usable capacity |
| **`allocation_ratio` ≠ 1** | Category ratios do not need to sum to 1.0 |
| **`type_count` grouping** | Multiple similar items can be grouped under one name |
| **Greedy fallback** | Works without SCIP installed; reduced optimality but same logic |

---

## Installation

```bash
# Required
pip install pyscipopt   # SCIP solver Python bindings

# Optional (prettier console output)
pip install tabulate
```

> If `pyscipopt` is not installed, the optimizer automatically falls back to a greedy algorithm.

---

## Usage

```bash
python stash_optimizer.py --inventory inventory.json
python stash_optimizer.py --inventory inventory.json --strategy greedy
python stash_optimizer.py --inventory my_inv.json --out results/output.json
```

| Argument | Default | Description |
|---|---|---|
| `--inventory` | `inventory.json` | Path to the input JSON file |
| `--strategy` | `scip` | `scip` or `greedy` |
| `--out` | same folder as inventory | Output JSON path |

---

## inventory.json Schema

```jsonc
{
  "stash_capacity": 1200,   // Total stash weight limit
  "buffer": 100,            // Weight to intentionally leave free
  "categories": [
    {
      "category": "Junk",
      "category_importance": 0.95,   // [0.0 - 1.0]
      "allocation_ratio": 0.40,      // Fraction of usable weight for this category
                                     // (does NOT need to sum to 1.0 across categories)
      "max_capacity": 500,           // Optional hard cap (overrides ratio if lower)
      "items": [
        {
          "name": "Loose screws",
          "type_count": 1,            // How many distinct sub-types share this entry
          "unit_weight": 0.03,        // Weight of a single unit
          "weight_modifier": 1.0,     // Multiplied with unit_weight (e.g. 0.5 with perks)
          "min_unit_quantity": 80,    // Minimum per sub-type (× type_count = min_total)
          "max_unit_quantity": 300,   // Maximum per sub-type (× type_count = max_total)
          "item_importance": 0.8,     // [0.0 - 1.0]
          "is_essential": true,       // Essential = stricter floor + steeper cut penalty
          "steal_factor": 0.6         // [0.0 - 1.0]  see Asymmetric Deviation Weights
        }
      ]
    }
  ]
}
```

### type_count explained

`type_count` lets you represent multiple similar items as a single entry.

```
type_count      = 15
unit_weight     = 0.1
max_unit_quantity = 3

→ max_total = 15 × 3  = 45 units
→ max weight = 45 × 0.1 = 4.5
```

### steal_factor semantics

| Value | Scarcity (being cut) | Surplus (expanding) |
|---|---|---|
| `1.0` | Hard to cut — resists losing space | First to expand — grabs available space |
| `0.0` | Easy to cut — sacrificed first | Last to expand — politely waits |

---

## Output

### Console table

```
══════════════════════════════════════════════════════
  Strategy=SCIP-v1.0  Usable=1100  Used=1099.82  (99.9%)
══════════════════════════════════════════════════════
─── Junk  importance=0.95  alloc=40%  target=440w  used=421w  fill=95.7%
  Loose screws   1   272  272   271.5  +1  [80..300]  190  300  -  -  0.030  8.16  0.60  1.00  ★
```

Columns: `Name · TC · Q/TC · TotalQ · Ideal · ΔBaseline · Range · PMin · PMax · d+ · d- · UnitW · TotalW · SF · Imp · Essential`

### optimization_result.json

```json
{
  "meta": {
    "strategy": "SCIP-v1.0",
    "stash_capacity": 1200.0,
    "buffer": 100.0,
    "usable_capacity": 1100.0,
    "total_weight_used": 1099.82,
    "total_weight_free": 0.18,
    "utilization_pct": 99.98
  },
  "categories": [
    {
      "category": "Junk",
      "importance": 0.95,
      "allocation_ratio": 0.4,
      "allocation_target": 440.0,
      "allocated_weight": 421.3,
      "allocation_fill_pct": 95.75,
      "items": [
        {
          "name": "Loose screws",
          "type_count": 1,
          "optimal_quantity": 272,
          "ideal_quantity": 271.493,
          "baseline_quantity": 271,
          "delta_from_baseline": 1,
          "min_total": 80,
          "max_total": 300,
          "preferred_min": 190,
          "preferred_max": 300,
          "deviation_positive": 0,
          "deviation_negative": 0,
          "qty_per_type": 272,
          "unit_weight": 0.03,
          "effective_weight": 0.03,
          "total_weight": 8.16,
          "is_essential": true,
          "steal_factor": 0.6,
          "composite_importance": 1.0
        }
      ]
    }
  ]
}
```

---

## Architecture

```
StashOptimizer          (orchestrator)
│
├── InventoryLoader     reads inventory.json → Category + InventoryItem objects
│
├── IdealCalculator     Phase 1: computes per-item target quantities
│   ├── _cascade()          per-category importance-weighted distribution
│   └── _global_rescale()   scales all ideals so Σ(ideal × weight) = usable
│
├── SCIPOptimizer       Phase 2: MIQCP single-pass solve
│   ├── _hard_lb()          absolute lower bound per item
│   ├── _hard_ub()          absolute upper bound per item (with surplus bonus)
│   └── optimize()          builds and solves the SCIP model
│
├── GreedyFallback      Phase 2 alternative when SCIP is unavailable
│
├── ReportRenderer      prints formatted console table
└── JSONExporter        writes optimization_result.json
```

---

## Optimization Model

### Phase 1 - IdealCalculator

Before the solver runs, the `IdealCalculator` computes a target (`ideal_quantity`) for every item. This gives the solver a well-informed starting point and prevents arbitrary solutions.

#### Step A — Per-category cascade

For each category with nominal budget `B_c`:

```
score_i  = composite_importance_i ^ α          (α = IMPORTANCE_ALPHA = 1.6)

initial allocation:
  alloc_i = (score_i / Σ score_j) × B_c / effective_weight_i
```

Items that hit their `max_total` return leftover budget to a pool. Items below `min_total` draw from the pool. This redistribution repeats up to `CASCADE_PASSES = 40` times until it converges.

#### Step B — Global rescale

After all categories finish their cascade, the global ideal weight may not equal `usable`. A single scale factor corrects this:

```
scale = usable / Σ(ideal_i × effective_weight_i)
ideal_i ← clamp(ideal_i × scale, min_total_i, max_total_i)
```

This ensures the solver is given a target that already aims for full capacity utilization.

#### composite_importance

```
composite_importance_i = 0.45 × category_importance + 0.55 × item_importance
                         + 0.15  (if is_essential, capped at 1.0)
```

---

### Phase 2 - SCIP Optimizer (MIQCP)

The optimizer solves a **Mixed-Integer Quadratically-Constrained Program**:

#### Decision variables

| Variable | Type | Description |
|---|---|---|
| `q_i` | Integer ≥ 0 | Quantity of item `i` |
| `give_c`, `take_c` | Continuous ≥ 0 | Budget a category gives / receives |
| `slack` | Continuous ≥ 0 | Unused capacity |
| `d_neg_i`, `d_pos_i` | Continuous ≥ 0 | Below-ideal and above-ideal deviation |
| `below_i` | Continuous ≥ 0 | How far item `i` falls below `min_total` |

#### Hard constraints

```
cap_hi :   Σ(effective_weight_i × q_i)  ≤  usable
cap_lo :   Σ(effective_weight_i × q_i)  ≥  usable × FILL_TOLERANCE   (= 0.95)

q_i    ≥   hard_lb_i
q_i    ≤   hard_ub_i
```

Where:

```
hard_lb_i = ESSENTIAL_HARD_FLOOR × min_total_i   (if essential)
          = 0                                      (if non-essential)

hard_ub_i = max_total_i                           (normal case)
          = max_total_i × (1 + bonus)             (if global surplus exists)
```

---

### Asymmetric Deviation Weights

This is the core mechanism for **smooth, directionally-correct spreading**.

The key insight is that `steal_factor` must behave differently depending on whether space is scarce or plentiful:

| Direction | High `steal_factor` means | Formula |
|---|---|---|
| **Scarcity** (cut below ideal) | Hard to cut — resists sacrifice | `w_neg = importance × steal_factor` |
| **Surplus** (grow above ideal) | Aggressive — grabs available space | `w_pos = importance / steal_factor` |

Both weight vectors are **normalized** across all items so the total penalty mass stays constant:

```
raw_neg_i = W_ITEM_DEV × composite_importance_i × steal_factor_i
raw_pos_i = W_ITEM_DEV × composite_importance_i / steal_factor_i

w_neg_i = (raw_neg_i / Σ raw_neg_j) × N
w_pos_i = (raw_pos_i / Σ raw_pos_j) × N
```

Normalization guarantees **smooth transitions**: raising one item's importance does not cause a sudden cliff in another item's allocation — it shifts the distribution gradually across all items.

The objective terms are:

```
penalty_dev = Σ_i [ w_neg_i × d_neg_i² + w_pos_i × d_pos_i² ]

where:
  d_neg_i = max(0,  ideal_i − q_i)   (how much below ideal)
  d_pos_i = max(0,  q_i − ideal_i)   (how much above ideal)
```

---

### Logarithmic Floor Protection

Items should be reluctant to fall below `min_total`, and must never collapse to zero.

A pure quadratic penalty `α × below²` has zero gradient at `below = 0`, which means the solver can drift items to zero without cost. The fix adds a linear term:

```
floor_penalty_i = α_i × (below_i² + below_i)

where  below_i = max(0, min_total_i − q_i)
```

At `below = 0`: gradient = `α` (non-zero — constant upward pressure).
As `below` grows: penalty accelerates quadratically.

```
α_i = W_FLOOR_ESS = 14.0   (if is_essential)
    = W_FLOOR_NON = 2.5    (otherwise)
```

Essential items face a **5.6× steeper** floor penalty, meaning they require much more capacity pressure to be cut below their minimum.

---

### Category Give / Take

Each category has a nominal budget `T_c = usable × allocation_ratio`. In practice, some categories may not fill their budget (e.g. weapons capped by `max_capacity`) while others need more. The give/take system allows **soft budget redistribution** between categories.

Variables `give_c` and `take_c` represent how much weight budget a category surrenders or receives. They are penalized **asymmetrically by importance**:

```
give_penalty_c = W_CAT_GIVE × (1 − importance_c)² × give_c²
take_penalty_c = W_CAT_TAKE × (1 − importance_c)  × take_c²
```

| Category importance | Give cost | Take cost | Behavior |
|---|---|---|---|
| Low (0.2) | Cheap `(0.8)²` | Expensive `(0.8)` | Gives readily, reluctant to take |
| High (0.9) | Expensive `(0.1)²` | Cheap `(0.1)` | Reluctant to give, takes freely |

A soft pool balance term penalizes imbalance between total gives and takes:

```
pool_balance_penalty = W_POOL_BAL × (Σ give_c − Σ take_c)²
```

This is kept **soft** (not a hard equality constraint) to preserve feasibility when integer rounding makes exact balance impossible.

---

### Surplus Relaxation

If the sum of all items at their `max_total` is still below `usable`, there is global surplus. In this case, items may exceed their `max_total`, scaled logarithmically by their importance:

```
hard_ub_i = ceil(max_total_i × (1 + bonus_i))

bonus_i   = log(1 + SURPLUS_LOG_BASE × surplus / usable) × composite_importance_i
```

More important items receive a larger bonus, so surplus space flows to the items that benefit most from it.

---

### Greedy Fallback

When `pyscipopt` is not available, a greedy algorithm runs instead:

**If over capacity → `_reduce()`:** For each item, compute a "victim weight":

```
victim_weight_i = (1 − category_importance_i)
                + (1 − composite_importance_i)
                + (1 − steal_factor_i)          ← high sf = hard victim = cut less
```

Cuts are distributed proportionally to `victim_weight × available_room × effective_weight`. Items with low importance and low steal_factor absorb the most.

**If under capacity → `_expand()`:** Items are sorted by `(category_importance, composite_importance × steal_factor)` — high steal_factor items expand first, filling available space aggressively.

---

## Tunable Constants

All constants are at the top of `stash_optimizer.py` and can be adjusted without touching the algorithm:

| Constant | Default | Effect |
|---|---|---|
| `IMPORTANCE_ALPHA` | `1.6` | Exponent on importance in cascade. Higher = more unequal distribution |
| `CASCADE_PASSES` | `40` | Max redistribution iterations in IdealCalculator |
| `FILL_TOLERANCE` | `0.95` | Minimum stash fill fraction (0.95 = 95%) |
| `W_ITEM_DEV` | `1.0` | Base scale for deviation penalty |
| `W_FLOOR_ESS` | `14.0` | Floor penalty for essential items below `min_total` |
| `W_FLOOR_NON` | `2.5` | Floor penalty for non-essential items below `min_total` |
| `W_EMPTY` | `60.0` | Penalty per unit of unused capacity |
| `W_CAT_GIVE` | `5.0` | Base cost for a category giving budget |
| `W_CAT_TAKE` | `3.0` | Base cost for a category receiving budget |
| `W_POOL_BAL` | `8.0` | Penalty for give ≠ take imbalance |
| `ESSENTIAL_HARD_FLOOR` | `0.80` | Essential items cannot go below `min_total × 0.80` |
| `SURPLUS_LOG_BASE` | `2.4` | Controls how far above `max_total` items can go on surplus |
| `MAX_TAKE_FRACTION` | `0.60` | Maximum extra budget a category can receive (60% of own budget) |

---

## What is SCIP?

**SCIP** (Solving Constraint Integer Programs) is one of the world's fastest open-source solvers for **mixed-integer programming** and **constraint programming**. It is developed at Zuse Institute Berlin (ZIB) and handles:

- **LP** — Linear Programming
- **MIP** — Mixed-Integer Programming (some variables must be integers)
- **QCP** — Quadratically-Constrained Programs (constraints with x² terms)
- **MIQCP** — Mixed-Integer Quadratically-Constrained Programs ← what this tool uses

SCIP finds the globally optimal solution (or proves none exists) by combining:

- **Branch and Bound** — splits the integer search space into a tree
- **Cutting planes** — adds linear constraints that tighten the feasible region
- **Presolving** — simplifies the model before solving
- **Primal heuristics** — finds good feasible solutions quickly

For this optimizer, SCIP handles the combination of integer item quantities (`q_i ∈ ℤ`) with quadratic objective terms (deviation penalties) that would be intractable to solve by enumeration.

More: [scip.zib.de](https://www.scipopt.org)

---

## What is PySCIPOpt?

**PySCIPOpt** is the official Python interface to the SCIP solver. It allows you to build optimization models using Python objects and call SCIP's solver without leaving Python.

```python
from pyscipopt import Model, quicksum

m = Model()
x = m.addVar(vtype="I", lb=0, ub=100, name="x")   # integer variable
y = m.addVar(vtype="C", lb=0.0,       name="y")   # continuous variable

m.addCons(x + y <= 50)                             # linear constraint
m.addCons(y >= x * x)                              # quadratic constraint (QCP)
m.setObjective(y, "minimize")
m.optimize()

print(m.getVal(x), m.getVal(y))
```

Key methods used in this optimizer:

| Method | Purpose |
|---|---|
| `model.addVar(vtype, lb, ub)` | Add a decision variable (`"I"` = integer, `"C"` = continuous) |
| `model.addCons(expr)` | Add a constraint (linear or quadratic) |
| `quicksum(...)` | Efficient sum of SCIP expressions (like `sum()` but for solver variables) |
| `model.setObjective(expr, "minimize")` | Set the objective function |
| `model.optimize()` | Run the SCIP solver |
| `model.getStatus()` | Check result: `"optimal"`, `"bestsolfound"`, `"infeasible"`, ... |
| `model.getVal(var)` | Read the solved value of a variable |

> **Note:** PySCIPOpt supports quadratic **constraints** (`x² ≤ y`) but the objective must be expressed through auxiliary variables. That is why this model introduces `td`, `tf`, etc. — they act as epigraph variables that linearize the quadratic objective terms into quadratic constraints.

Install: `pip install pyscipopt`  
Docs: [pyscipopt.readthedocs.io](https://pyscipopt.readthedocs.io)

---

## License

MIT
