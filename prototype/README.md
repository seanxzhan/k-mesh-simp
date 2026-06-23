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

  --- skinning: drive the high-res "visual" mesh from the coarse proxy sim ---
  harmonic_skinning.py   the harmonic prolongation  S = -K_ee^{-1} K_er  (+ geometric baseline)
  viz_lbs.py             four decimation-time skinning maps, painted + driven (--weights)
  eval_sim_harmonic.py   full pipeline: PBD-sim proxy -> drive fine mesh (corotational, --relax)
  check_drift.py         quantify placement drift / proximity scrambling of the id-correspondence
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

## Skinning — driving the high-res mesh from the coarse proxy

Decimation gives us a **coarse proxy** (~128 verts) that is cheap to simulate (PBD). But we
still want to *display* the original **fine mesh** (~1300 verts). So we need a rule that
takes the proxy's motion each frame and produces the fine mesh's motion. That rule is a
**skinning map** (in multigrid/FEM language, a **prolongation** — an operator that
"prolongs" a coarse field onto a fine one).

This section explains, from the ground up, the map we use, the three things that went wrong
with the naive version, and the fixes. **The recipe we settled on:** the harmonic map
`S = -K_ee⁻¹ K_er`, applied as **co-rotational LBS** (per-handle rotation) with **soft-handle
smoothing** (`--relax`). Everything here is exercised by `eval_sim_harmonic.py`.

### Background: Linear Blend Skinning, and where the weights come from

If you've seen character rigging, you've seen **Linear Blend Skinning (LBS)**: each mesh
vertex follows a weighted average of a few "bones". Here the "bones" are the coarse proxy
vertices, which we call **handles**. If handle `k` moves by displacement `d_k`, then fine
vertex `i` moves by

```
d_fine[i] = Σ_k  w[i,k] · d_k ,        with   Σ_k w[i,k] = 1   ("partition of unity")
```

The weights `w[i,k]` say how strongly handle `k` drags vertex `i`. The partition-of-unity
condition (rows sum to 1) guarantees that if *all* handles translate by the same `t`, the
whole fine mesh translates by `t` — no shrinking or drifting.

The entire question is: **where do the weights come from?** Common answers are *paint them by
hand* (artists) or *inverse distance to the nearest handles* (geometric). Our pitch is
different: **read them off the physics** — the same elastic operator `K` that drove the
decimation.

### The harmonic map  S = -K_ee⁻¹ K_er  (energy-minimizing skinning)

**Intuition first.** Picture the mesh as a web of springs (edge springs that resist
stretching + hinge springs that resist bending). If you displace all vertices by a stacked
vector `d`, the stored elastic energy is

```
E(d) = ½ dᵀ K d
```

`K` is the **stiffness matrix**: it encodes how hard each deformation is. Stiff directions
cost lots of energy; floppy directions cost little. (Rigidly translating or rotating the
*whole* mesh costs **zero** energy — those are `K`'s null space.)

Now **pin the handles** at prescribed displacements and let every other vertex settle into
the **lowest-energy shape** consistent with those pins — exactly like a rubber sheet tacked
down at a few points sagging into its most relaxed configuration. That relaxed shape *is* our
skinning. To compute it, split the vertices into **retained** handles `r` and **eliminated**
interior `e`, and block the matrix:

```
E = ½ [d_e]ᵀ [K_ee  K_er] [d_e]
       [d_r]  [K_re  K_rr] [d_r]
```

Minimize over the free vertices `d_e` (set the derivative to zero):

```
∂E/∂d_e = K_ee d_e + K_er d_r = 0     ⟹     d_e = -K_ee⁻¹ K_er d_r
```

Stack the identity for the handles themselves and you get the full coarse→fine map:

```
d_fine = S d_coarse ,      S = [        I        ]   ← handle rows (each handle follows itself)
                               [ -K_ee⁻¹ K_er ]      ← interior rows = the skinning weights
```

The rows of `-K_ee⁻¹ K_er` **are** the skinning weights, in closed form, straight from the
material — no painting, no fitting to animation.

**Two names for the same object** (useful for reading papers):
- **Harmonic extension.** For the simplest energy (`∫|∇u|²`, the membrane/Laplacian energy),
  this is literally solving Laplace's equation `Δu = 0` with the handles as boundary values —
  a *harmonic* function. Our `K` is a richer mechanical energy, but the idea is identical.
- **Schur complement / static condensation (Guyan reduction).** Eliminating the interior
  DOFs and keeping only the handles is the Schur complement of `K_ee`; structural engineers
  call it static condensation. It preserves the *static* stiffness exactly while shrinking
  the DOF count.

Because `K` annihilates rigid motion, `S` reproduces a rigid motion of the handles *exactly*
(this is why the weights sum to 1). `harmonic_skinning.py` builds `S` and checks this; on
elastic test deformations it beats an inverse-distance kNN baseline, because it is optimal in
the *elastic-energy* norm — the regime a simulated proxy actually lives in.

### Two ways to apply the map: full 3×3 block vs scalar LBS

`S` is really a `3n × 3n_coarse` matrix of small **3×3 blocks** `S_block[i,k]` (x/y/z are
coupled through `K`). There are two ways to use it:

- **Full 3×3 block:** `d_fine[i] = Σ_k S_block[i,k] · d_k`. The *off-diagonal* block entries
  let a handle's x-motion induce some y/z motion in a fine vertex — and that cross-coupling
  is exactly what lets the map represent **rotation**.
- **Scalar LBS:** collapse each block to a single number `w[i,k] = ⅓·tr(S_block[i,k])` (rows
  still sum to 1) and apply it identically to x/y/z. This is the *textbook* LBS form — one
  weight per influence — and it is what game engines and our sister project `proxy-asset-gen`
  consume. Simpler and sparser, but **translation-only**: no rotation coupling.

`prolongation_scalar_weights()` does the trace reduction; `viz_lbs.py` paints both and shows
the deformation each produces. (Our scalar `W` is ~90% dense but only ~3.2 *effective*
handles per vertex, so a top-k≈8 sparsification matches the engine/`proxy-asset-gen` format
with little loss.)

### Three things that go wrong — and the fixes

The naive "full block, pinned handles" map looks bad under a real swinging sim. There are
three distinct causes; each has a clean, controlled fix.

**1. Drift (the proxy isn't where the original vertex was).** Our decimation slides each
surviving coarse vertex to a *stiffness-optimal* rest position `C[k]`, which is **not** the
original location `F[survivors[k]]` of the fine vertex it came from (measured drift on the
skirt: ~1.4× edge length on average, up to ~5×). But the map's handle rows are the *identity*
— they silently assume handle `k` sits exactly on fine vertex `survivors[k]`. So a handle
*translation* still transfers perfectly, but a handle *rotation* transfers with an error
proportional to `drift × angle`. (`check_drift.py` quantifies this; ~45% of handles drift
farther than the gap to their nearest neighbor.)

**2. Rotation → inflation.** `K` is the stiffness at the *rest* pose — a **linearized**
(small-displacement) operator. Feed a *finite* rotation through a linear operator and it
reads as a *stretch*: a point at radius `r` rotated by angle `θ` is sent to `r·√(1+θ²)` — it
**inflates** (~40% at θ ≈ 1 rad). So a skirt swinging 30–60° balloons. (Scalar LBS is worse:
being translation-only, it cannot rotate at all.)

> **Fix — co-rotational LBS.** Keep the operator weights, but give each handle a *finite*
> rotation `R_k`, estimated from how its coarse 1-ring deformed (the **ARAP / polar-
> decomposition** trick: SVD of the rest-vs-current edge cross-covariance). Apply
> ```
> x_fine[i] = Σ_k  w[i,k] · [ X_k + R_k (F_i − C_k) ]
> ```
> Now the rotation is applied *exactly* (no inflation), and the drift offset `(F_i − C_k)` is
> carried *rigidly* under `R_k`, so drift stops scrambling things. Verified by a built-in
> diagnostic: under a synthetic 30° proxy tilt, co-rotational reproduces it to **~0** error,
> while the full block is off by **18%** and scalar LBS by **17–25%**.
>
> The lesson: **operator-faithful weights and visually-good motion are NOT mutually
> exclusive.** The conflict was only that a rest-pose *linear* operator cannot represent big
> rotation. Co-rotational *decouples* the two jobs — the operator decides *which handles
> influence a vertex* (the weights, which carry the mechanics); the per-handle rotation
> handles the *large rotation* (the visuals).

**3. Kinks at the handles.** The harmonic map *interpolates*: its handle rows are the
identity, so each fine handle vertex is pinned **exactly** onto its (faceted, drifted) proxy
vertex. A hard point constraint that pokes off the smooth trend creates a **gradient cusp** —
a visible kink — right at every handle. (This is the classic signature of a membrane/Laplacian
interpolant: minimizing `∫|∇u|²` through point constraints leaves a cusp at each point, like
`|x|` in 1D or `log r` in 2D.) Two attempts:

- **More bending (`--smooth`) — tried, doesn't fix it.** A *bending* (thin-plate) energy
  `∫|Δu|²` is C¹-smooth, so blending bending into `K` smooths the surface *between* handles.
  But the cusp comes from a *hard constraint*, not from the interior operator — and making the
  interior flatter just makes the pinned handle poke out *more* (measured: it can make things
  slightly worse). Pure bending *does* remove the cusp but then **overshoots** (rings, goes
  negative) — which is exactly why "Bounded Biharmonic Weights" need extra inequality
  constraints.
- **Soft handles (`--relax`) — this is the fix.** Stop pinning. Instead of forcing
  `d_fine[handle] = d_proxy`, attach a *spring* and minimize
  ```
  ½ dᵀ K d  +  (λ/2) · |d_handle − d_proxy|²
  ```
  giving `S = λ (K + λ D_h)⁻¹ P_hᵀ` (`D_h` = handle-DOF selector). The handle is now pulled
  *toward* the proxy but allowed to relax into the smooth surface, so the cusp becomes a gentle
  bump. One knob `λ` (exposed as `--relax R`, `λ = median(diag K)/R`): larger = softer =
  smoother, at the cost of a little handle "slip". Measured handle-roughness (graph-Laplacian
  energy at handles ÷ mesh average; 1.0 = no kink): co-rotational drops **1.77 → 1.17** at
  `R=20`, with **no overshoot** and rigid motion still reproduced exactly.

`eval_sim_harmonic.py` prints both diagnostics every run — the **finite-rotation fidelity**
table (problem 2) and the **handle-kink roughness** ratio (problem 3) — so you can tune the
knobs against numbers, not just eyeballs.

### The recipe we settled on

```
  S = -K_ee⁻¹ K_er                  harmonic prolongation (weights from the material)
   → w = ⅓·tr(S_block)              scalar LBS weights (engine-friendly form)
   → co-rotational application       per-handle R_k  → fixes rotation + drift  (problem 1,2)
   → soft handles (--relax)          approximate, don't pin → fixes kinks      (problem 3)
```

```bash
python prototype/eval_sim_harmonic.py --deform corotational --relax 20   # the settled recipe
python prototype/eval_sim_harmonic.py                                     # full | scalar | corotational
python prototype/eval_sim_harmonic.py --smoke --frames 12 --settle 8      # headless checks + diagnostics
python prototype/viz_lbs.py                                               # paint the four decimation-time maps
```

### What we tried (and kept / rejected)

| Approach | Idea | Verdict |
|---|---|---|
| geometric kNN weights | bind to nearest handles by inverse distance | baseline; ignores the material |
| **global harmonic `S` (full 3×3)** | one-shot energy-minimizing extension | smooth & reproduces rotation *in theory*; drift + linearization hurt in practice |
| local harmonic (per-collapse) | accumulate `-K_vv⁻¹K_vj` during each collapse | bounded support; an approximation to the global Schur extension |
| edge blend / mean-value coords | stiffness-free baselines (`viz_lbs --weights edge/geom`) | piecewise-constant / purely geometric reference points |
| **scalar LBS** (`⅓·tr`) | engine-standard one-weight-per-influence form | simplest, sparsifiable; translation-only |
| **co-rotational** | per-handle finite rotation `R_k` | ✅ **kept** — fixes rotation & drift |
| `--smooth` (more bending) | bending-dominated operator → C¹ interior | ✗ rejected for kinks — can't soften a hard constraint (and pure bending overshoots) |
| **soft handles `--relax`** | approximate the proxy, don't pin it | ✅ **kept** — fixes the handle kinks, no overshoot |
| snap coarse→fine | move the proxy onto its fine anchors to zero the drift | ✗ rejected & removed — destroys the stiffness-optimal positions; co-rotational *absorbs* the drift instead |

### References (for the curious student)

- *Schur complement / static condensation:* Guyan, "Reduction of stiffness and mass
  matrices" (1965).
- *Harmonic & biharmonic skinning weights:* Jacobson, Baran, Popović, Sorkine, **"Bounded
  Biharmonic Weights for Real-Time Deformation"** (SIGGRAPH 2011) — why biharmonic is smooth
  but needs bounds to avoid overshoot.
- *As-Rigid-As-Possible / polar decomposition:* Sorkine & Alexa, "As-Rigid-As-Possible Surface
  Modeling" (2007) — the per-handle rotation estimate.
- *Co-rotational FEM:* Müller & Gross, "Interactive Virtual Materials" (2004).

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
