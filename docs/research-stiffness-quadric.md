# Research: Stiffness Quadric

## Central Question

Can we construct a "stiffness quadric" Q_v^K that:
1. Encodes mechanical error (not just distance to planes)
2. Accumulates additively on collapse (Q_w = Q_u + Q_v)
3. Evaluates without reference to the current mesh state (O(1) per edge)
4. Produces physically-informed skinning weights as a byproduct

## Why This Matters

QEM accumulates because its error (distance² to planes) is a quadratic form in position alone with fixed coefficients. Operators (Laplacian, stiffness) break this because their cost involves the current mesh state. Nobody has shown how to fold an operator's influence into a purely position-dependent quadratic form that accumulates.

Solving this gives simulation-aware mesh simplification that is as fast and clean as QEM, while producing proxy meshes optimized for elastic simulation and LBS reconstruction — replacing the post-hoc skinning weight optimization in proxy-asset pipelines (Zheng et al. 2024).

## Context Explored So Far

- Schur complement accumulates mechanically but is geometry-blind
- Strain energy captures geometry but doesn't accumulate (operator depends on current state)
- Spectral simplification (Lescoat 2020) has partial accumulation (signals through P) but snapshot operator evaluation
- Nakagawa & Kanai 2026 use modal strain as a static weighting on QEM (no accumulation, no skinning weights)
- The restriction matrix P built during simplification IS the skinning weight matrix

## Why QEM's Quadric Works

- E(x) = x^T A x + 2b^T x + c — error is a function of position x only
- A, b, c are computed from the original geometry (fixed forever)
- A encodes normals (nn^T), b encodes signed distances, c encodes constant offsets
- On collapse: Q_w = Q_u + Q_v (just add matrices)

## What We Need

A similar (A, b, c) that encodes "mechanical importance of this vertex's position" rather than "distance to tangent planes."

---

## Approach 1: Stiffness-Weighted Plane Quadric

The simplest extension: weight each face's plane quadric by its stiffness contribution rather than just area:

```
Q_v = Σ_faces  stiffness_weight_f × plane_quadric_f
```

where `stiffness_weight_f` comes from the element stiffness magnitude (e.g., Frobenius norm of the 9×9 element stiffness matrix for that face).

**Properties:**
- Still accumulates (it's just differently-weighted QEM)
- Stiff faces "count more" — their geometry is more expensive to violate
- O(1) per edge evaluation, same as QEM

**Limitations:**
- Shallow — uses stiffness only as a scalar weight, doesn't encode the operator structure
- Doesn't capture off-diagonal coupling (how stiffness flows between vertices)
- Vertex positioning is still purely geometric (minimize distance to stiffness-weighted planes)

---

## Approach 2: Strain Energy Quadric (Linearized)

For small displacements δ from original position x₀, the strain energy is:

```
U(x₀ + δ) ≈ ½ δ^T K_vv δ
```

where K_vv is the 3×3 diagonal block of the original stiffness matrix at vertex v. Expanding in terms of x = x₀ + δ:

```
U(x) = ½ x^T K_vv x - x^T K_vv x₀ + ½ x₀^T K_vv x₀
```

This IS a quadratic form: A = K_vv, b = -K_vv x₀, c = ½ x₀^T K_vv x₀.

**Properties:**
- Accumulates: K_vv folds through the Schur complement on collapse. After eliminating v, u's effective self-stiffness increases by the Schur correction K_BA K_AA⁻¹ K_AB. This is additive.
- Material-aware: stiff vertices (high K_vv eigenvalues) are penalized more for displacement
- Position-dependent: cost depends on how far x deviates from the original position x₀
- O(1) per edge evaluation

**What it means geometrically:** "Don't move stiff vertices." A vertex with high K_vv resists displacement — if you place the merged vertex far from x₀, the strain energy is high. The stiffness matrix naturally encodes which directions are expensive (anisotropic if the local geometry is anisotropic).

**Key question:** Does K_vv alone give meaningful vertex placement? It penalizes displacement from original position weighted by stiffness — but it doesn't know about the surface (no tangent planes). A vertex could be placed off-surface if K_vv happens to have a low eigenvalue in the normal direction.

**Accumulation mechanism:**
- Eliminate v: K_uu^new = K_uu + K_uv K_vv⁻¹ K_vu (Schur correction)
- The new quadric for u has A = K_uu^new, b = -K_uu^new x_u, c = ½ x_u^T K_uu^new x_u
- This naturally increases u's "resistance to movement" as it absorbs v's mechanical role

---

## Approach 3: Combined Plane + Stiffness Quadric

```
Q_v = Q_plane + λ · Q_stiffness
```

where:
- Q_plane = standard QEM quadric (stay near original tangent planes)
- Q_stiffness = (K_vv, -K_vv x₀, ½ x₀^T K_vv x₀) from Approach 2

**Properties:**
- Accumulates: both terms are additive quadrics. Q_plane accumulates as in QEM, Q_stiffness accumulates via Schur complement.
- Captures both geometric fidelity (planes) and mechanical importance (stiffness)
- λ controls the tradeoff: λ=0 is pure QEM, λ→∞ is pure stiffness
- Can't be worse than QEM (it strictly adds information)
- O(1) per edge evaluation

**The balance λ:**
- Should be dimensionally consistent (QEM has units of length², stiffness quadric has units of force×length)
- Could normalize by total mesh area / total stiffness trace
- Or: set λ so that median(Q_stiffness cost) ≈ median(Q_plane cost) across edges

**Accumulation on collapse (u,v) → w:**
- Q_plane_w = Q_plane_u + Q_plane_v (standard QEM addition)
- Q_stiffness_w: need to merge stiffness quadrics. The Schur complement gives K_uu^new, but x₀ for the merged vertex needs to be defined. Options:
  - Use the QEM optimal position as the new x₀
  - Use the original position of u (the kept vertex)
  - Use the restriction-matrix-weighted blend of original positions

---

## Next Steps

- Prototype Approach 2 in a notebook to see what K_vv as a quadric produces on a real mesh
- Prototype Approach 3 and compare against pure QEM and QEM+flow
- Investigate whether the Schur-accumulated K_vv gives meaningful vertex placement or needs the plane term to stay on-surface
