# Mechanical-QEM cost prototype

A clean restart implementing the costs in [`docs/mech_qem.tex`](../docs/mech_qem.tex).
Self-contained; ignores the earlier exploratory modules (`schur.py`,
`stiffness_quadric.py`, `simplify_schur.py`, `simplify_stiffness_quadric.py`).

```
prototype/
  mech_qem.py            core math: per-element K_e, affine probe, G_e, accumulation, costs
  viz_costs.py           polyscope viz of the cost fields + headless --smoke validation
  simplify_mechanics.py  greedy edge-collapse decimation driven by the mechanical cost
  viz_simplify.py        compare mechanical vs QEM decimation (polyscope + --smoke)
```

## Run

```bash
python prototype/viz_costs.py                 # dress/skirt mesh, interactive polyscope
python prototype/viz_costs.py --smoke         # headless: runs all correctness checks
python prototype/viz_costs.py --mesh data/spot.obj
python prototype/viz_costs.py --thickness 0.02     # thicker shell => more bending
python prototype/viz_costs.py --no-collapse-cost   # skip Layer B (faster)

python prototype/viz_simplify.py                   # decimate skirt -> 600v, compare to QEM
python prototype/viz_simplify.py --target 400      # heavier decimation
python prototype/viz_simplify.py --geom-weight 2.0 # heavier visual (geometric) term
python prototype/viz_simplify.py --smoke           # headless verification
```

## What is implemented (maps 1:1 to the doc)

**Phase 0 — assemble the mechanical model, per *element* (not the global matrix):**
- CST membrane triangle `K_e ∈ R^{9×9}`, rank 3.
- discrete bending hinge `K_h ∈ R^{12×12}`, rank 1.

**Phase 1 — probe subspace `V` = the 6 affine / constant-strain modes** (the doc's
default fork). A node's affine displacement factors through its rest position:
`d_n = F X_n + c = P(X_n) a`, `a = (vec F, c) ∈ R^12`.

**Phase 2 — project + accumulate:**
- `G_e = V_eᵀ K_e V_e ∈ R^{12×12}` per element.
- `G_v = Σ_{e∋v} G_e` per vertex (QEM-style accumulation).
- `G = Σ_e G_e` global (PSD, **effective rank 6** — verified).

**Costs (two layers, both visualized):**

| Layer | What | Where | Meaning |
|---|---|---|---|
| **A** | probe energy of `G_e` / `G_v` | per-triangle, per-vertex | *response template* — "how much homogenized response lives here". **Not** an error (see doc divergence 2). |
| **B** | `‖G_after − G_fine‖²_W` over affected elements, min over candidate placements, reduced to per-vertex via min incident edge | per-edge → per-vertex | the **actual decimation cost** — "cheapest edge collapse here". |

Visualized quantities (one mesh, toggle in the polyscope panel):
- faces: `tri area`, `tri membrane cost`, `tri |Ke|_F` (sliver indicator)
- vertices: `vtx membrane / bending / total cost`, `vtx min collapse cost (+log10)`

## Validation (`--smoke`, all pass)

- **CST patch test**: uniform strain stores exactly `½ A t εᵀ D ε` (rel err ~3e-15).
- **Rigid invariance**: membrane & hinge annihilate all 6 rigid modes.
- **`G` is PSD with nullity exactly 6** — the doc's "effective rank 6" (null space =
  3 translations + 3 infinitesimal rotations).
- **Doc sanity check 1**: per-triangle membrane cost is *exactly* proportional to area
  (`cost = c·area`, std/mean ~1e-14, `c = ½ t·tr(D)`). I.e. the affine-probe membrane
  quadric reduces to `D`-reweighted geometric QEM, as the doc predicts.
- Layer B collapse costs finite and ≥ 0.

## Decimation (Phase 3) — `simplify_mechanics.py`

A greedy edge-collapse decimator that *uses* the Layer B cost. Structure mirrors
`kms.simplify_qem` (a `MeshAdjacency` topology engine + a timestamped lazy heap); the
only structural change is that the mechanical cost depends on **geometry**, not on a
frozen accumulated quadric — so each collapse changes the cost of every edge in the
merged 1-ring, and that whole neighborhood is re-pushed (QEM only re-pushes edges
incident to the merged vertex).

```python
from simplify_mechanics import simplify_mechanics
coarse = simplify_mechanics(mesh, target_verts=400, thickness=1e-3,
                            geom_weight=0.0)   # 0 = pure mechanical; >0 blends QEM
```

- **cost** = `‖G_after − G_fine‖²_F` over affected elements (the doc's cost fork (a), `W = I`).
- **placement** (`placement="quadratic"`, default): the merged vertex is constrained to
  the edge, `w(α) = (1-α)u + α v`. We sample the cost at `α ∈ {0, 0.5, 1}`, fit a parabola
  `p(α)`, and place at its constrained minimizer `α* ∈ [0,1]` — an on-edge 1-D line-search
  (a true 3-D `-A⁻¹b` solve isn't available: the cost is not quadratic in a free position).
  We keep the best (by true cost) of `{0, 0.5, 1, α*}`, so it never does worse than the old
  3-point min, and a degenerate placement self-rejects (zero area → `ΔG` jumps). On-edge
  placement also preserves the surface-sticking property. `placement="endpoints"` recovers
  the old behavior.
- **`geom_weight`** blends a geometric QEM term (the `α·E_geom` fork); both terms are
  normalized by their initial medians so the weight is dimensionless.
- **`probe` + `bending_weight`** turn on the bending term (see below); default is
  affine + membrane-only (~20 ms/collapse; bending-aware curvature ~130 ms/collapse).

**Cost normalization (so the weights make sense).** The collapse cost is a weighted sum
of three terms, each divided by the **median of its own initial per-edge cost**:

```
cost = membrane/mem_scale  +  bending_weight · bending/bend_scale  +  geom_weight · geom/geom_scale
```

`membrane = ‖ΔG_membrane‖²_F`, `bending = ‖ΔG_bending‖²_F`, `geom` = QEM error. Because
each term is normalized to its own typical magnitude, **`bending_weight` and `geom_weight`
are dimensionless and on the same footing** — a weight of `1` means "as costly as a typical
membrane collapse". Useful range ~`0`–`10` (0 = off, ~1 = balanced, ≥4 = that term
dominates). Normalizing bending also divides out its physical `~t³` magnitude, so the knob
is **thickness-independent**. (The terms are penalized separately rather than as one
assembled `‖ΔG‖²`; the dropped cross term is small because membrane and bending excite
near-disjoint modes.)

Observed on the skirt at 400v (`viz_simplify.py`): pure-mechanical keeps a **larger
minimum triangle area** than QEM (it dislikes destroying stiff triangles) but can leave a
few higher-aspect slivers; **mechanical + geom (α=1)** is the cleanest on every quality
metric (lowest area ratio, lowest max aspect). This is the doc's point made concrete:
the mechanical term alone is not a quality term, so `E_geom`/`E_triq` earn their place.

### Making bending matter (curvature probe + bending in the collapse cost)

Two pieces are required, both implemented:

1. **Bending in the collapse cost** (`bending_weight > 0`). The local hinges are
   re-evaluated *before vs after* each candidate collapse — necessary because a collapse
   moves the dihedral reference of every edge in the 1-ring (the doc's fragile part). It
   enters as its own normalized term `bending_weight · ‖ΔG_bending‖²/bend_scale`.
2. **Curvature-enriched probe** (`probe="curvature"`). Affine fields have zero curvature,
   so the hinge is ~null under them (verified: flat-grid bending energy `2e-17` affine vs
   `1e-4` curvature). Adding the 18 constant-curvature modes (`a ∈ R^30`) lets bending
   register. Still factors through node position, so the per-vertex additive reduction and
   the rank-6 affine null space survive (verified on the skirt). `bending_weight > 0`
   should be paired with `probe="curvature"` (viz_simplify does this automatically).

Because bending is normalized, `bending_weight ≈ 1` already preserves features and the
effect saturates by ~4 (no thickness tuning needed). Demonstration (`--crease-demo`, a
folded grid): bending-aware keeps a **clean, sharp crease** (crease vertices `6 → 14`,
min_area `1.3e-3`, max aspect `1.8`) where **membrane-only flattens it and produces
zero-area faces**. Note the physical fact remains that for a thin shell bending stiffness
is `~t²` of membrane — normalization is precisely what lets you weight bending up to where
it influences the decimation regardless.

## ⚠️ kms.stiffness bugs found (and fixed here)

While building Phase 0 I found **two independent bugs in `src/kms/stiffness.py`** that
break the foundational FEM. This prototype does **not** import that element math; it
uses corrected formulas, and `--smoke` prints the discrepancy as a diagnostic.

1. **CST `B`-matrix sign error** (`membrane_stiffness_cst`). Row 0 (`εxx`) is negated and
   the two `y2` entries of row 2 (`γxy`) are flipped vs. the standard CST. Consequences:
   - the Poisson coupling `εxx·εyy` is silently **negated**;
   - the element **fails the patch test** (stores the wrong energy under uniform strain —
     off by ~60% on a test triangle);
   - it does **not** annihilate in-plane rigid rotation.

2. **Hinge gradient not rigid-invariant** (`bending_stiffness_hinge`). The analytic
   dihedral-angle gradient violates translation invariance badly
   (`‖Σ gᵢ‖ / ‖g‖ ≈ 1`, should be 0) and correlates only `-0.99` with the
   finite-difference gradient of the true dihedral angle. It does not annihilate rigid
   body modes, so the assembled operator loses its rank-6 affine null space.

This prototype fixes (1) with the standard `B` and (2) by taking `g = ∇θ` via central
finite differences of a (rigid-invariant) dihedral angle — provably correct, with an
analytic gradient as a trivial later optimization. **Recommend fixing `kms/stiffness.py`
itself** (it likely also affected the abandoned schur / stiffness-quadric experiments).

## Forks on the road

The doc's explicit fork table, annotated with **what this prototype picked** and what's
**still open**:

| Fork | Options | Doc lean | Prototype | Open? |
|---|---|---|---|---|
| **probe subspace** | affine / +curvature / +eigenmodes / +loads | affine first | **affine (6 modes)** | ⬅ the deepest fork — this *defines* "responsiveness" |
| **cost backbone** | (a) additive `‖G_after−G_fine‖` vs (b) exact differential-mode Schur per collapse | (a), validate vs (b) | **(a)** | validate (a) vs (b) on a 1-ring (doc check 2) |
| **bending bookkeeping** | fold into assembled `G_e` vs distribute to hinge's 4 verts | assembled | **accumulated to 4 verts (Layer A)** | bending is **not yet in Layer B** (collapse cost) |
| **merged placement** | closed-form min of `G`-deviation vs candidate set | candidate set | **on-edge quadratic line-search** (`w(α)`, fit `p(α)` at α∈{0,.5,1}, min on [0,1]) | global free-3-D solve is non-quadratic; greedy makes per-step gains path-dependent |
| **metric `W`** | identity / energy-weighted / mode-importance | identity | **identity** | open |

Conceptual forks the doc raises that are worth deciding deliberately:

- **Probe = definition of responsiveness.** Affine preserves *homogenized membrane
  stiffness* (the patch test) and — as verified — is ~geometric QEM reweighted by `D`.
  To preserve **bending / global compliance / specific loads** you must enrich the probe
  (curvature modes, low-frequency eigenmodes, or a load subspace). Biggest design lever.
- **Affine is (nearly) blind to bending.** Under the affine probe the bending field is
  tiny (curvature-coupling only). Preserving bending *response* requires constant-curvature
  (quadratic) probe modes. So the current bending field is a diagnostic, not a cost driver.
- **Bending must be re-evaluated at the collapse step** (not frozen): collapsing one edge
  perturbs the dihedral reference of every edge in the 1-ring. This is exactly why Layer B
  is membrane-only for now — adding bending means re-meshing hinges per candidate collapse.
- **`G_e(x*)` is not exactly quadratic in `x*`** (moving the rest vertex changes `B`, area,
  `K_e` rationally/with square roots), so there's no QEM-style closed-form `-A⁻¹b`. Hence the
  on-edge quadratic line-search (`placement="quadratic"`): sample 3 points on the edge, fit a
  parabola, take its minimizer. It's a per-collapse improvement, but greedy decimation is
  path-dependent — lowering each edge's cost reorders the heap, so the *cumulative*
  homogenized-stiffness fidelity isn't monotonically improved (skirt→600v: deviation
  9.7e-4 quadratic vs 8.6e-4 endpoints, both ~0.1%; max aspect 10.7 vs 12.4).
- **Dirty meshes / spurious null modes.** Floppy or dangling structure has zero-energy
  modes, so the *mechanical* cost of deleting it is ~0 — the decimator would happily delete
  it. Real assets therefore need a **geometric/feature term `α·E_geom`** (protect
  visually-important but mechanically-free structure) and a **sliver term `γ·E_triq`**
  (required for a *simulatable* output). Neither is in this v1. Non-manifold edges also need
  a hinge convention. (This skirt is clean: 1 component, 144 boundary/hem edges, no
  non-manifold or degenerate faces — so these don't bite *here*, but will on full garments.)

## Not done yet (natural next steps)

- A principled sliver term `E_triq` (pure-membrane can still make zero-area faces on
  sharp creases — see the crease demo warning) and a proper `E_geom` weighting schedule.
- Cross-validation of cost backbone (a) against the exact differential-mode Schur (b)
  on a single 1-ring (the doc's correctness check 2).
- Eigenmode / load-subspace probes (the curvature probe is in; these are the remaining
  probe enrichments that dial toward Li-style mode-shape attribute quadrics).
- Bending in the Layer B *viz field* (`mech_qem.membrane_collapse_costs`) — it's in the
  decimator now, but `viz_costs.py`'s collapse-cost field is still membrane-only.
- Speed: the bending-aware curvature path is ~130 ms/collapse (analytic hinge already in;
  the curvature `_stack_P` and 30×30 projections are the next hot spot to vectorize).
