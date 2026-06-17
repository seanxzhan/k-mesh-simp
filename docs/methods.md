# Stiffness-Informed Mesh Simplification: Methods

This document explains the four stiffness-based simplification methods implemented in this repo. Each builds on the previous, addressing its limitations.

## Background: The Problem

Given a high-resolution triangle mesh (the "visual mesh"), produce a coarse mesh (the "proxy") suitable for physics simulation. The proxy should:

1. Simulate efficiently (few vertices)
2. Reproduce the fine mesh's deformation behavior when driven back via skinning
3. Produce skinning weights as a byproduct (no post-hoc optimization)

The standard approach (QEM) optimizes for geometric fidelity — distance to the original surface. We want to optimize for **mechanical fidelity** — how well the coarse mesh preserves the fine mesh's elastic response.

## Background: How QEM Works

QEM (Quadric Error Metrics, Garland & Heckbert 1997) simplifies meshes by greedily collapsing the cheapest edge, where "cheap" means "minimal geometric error."

### The Key Observation

In the original model, each vertex is the solution of the intersection of a set of planes — the planes of the triangles that meet at that vertex. The error of placing a vertex at position v with respect to this set of planes is the sum of squared distances:

```
Δ(v) = Σ_{p ∈ planes(v)}  (pᵀv)²
```

where p = [a b c d]ᵀ represents the plane ax + by + cz + d = 0 (with a² + b² + c² = 1), and v = [x y z 1]ᵀ in homogeneous coordinates.

### The Fundamental Quadric

This error metric can be rewritten as a quadratic form:

```
Δ(v) = Σ_p (vᵀp)(pᵀv) = Σ_p vᵀ(ppᵀ)v = vᵀ (Σ_p Kp) v
```

where Kp = ppᵀ is the 4×4 **fundamental error quadric** for plane p:

```
        ┌                 ┐
        │ a²  ab  ac  ad  │
Kp =    │ ab  b²  bc  bd  │
        │ ac  bc  c²  cd  │
        │ ad  bd  cd  d²  │
        └                 ┘
```

This single matrix Kp encodes the squared distance from any point in space to plane p. Summing these over all incident planes gives a single matrix Q = Σ Kp that represents an entire set of planes.

### Per-Vertex Quadric

Each vertex accumulates the fundamental quadrics from its incident faces (area-weighted):

```
Q_v = Σ_{faces f ∋ v}  (area_f / 3) · K_{plane(f)}
```

At the original vertex position, Δ(v₀) ≈ 0 (the vertex lies on all its tangent planes).

### Edge Collapse and Accumulation

For edge (u,v), the merged quadric is simply:

```
Q_merged = Q_u + Q_v
```

This is equivalent to `planes(w) = planes(u) ∪ planes(v)` — the new vertex inherits ALL tangent plane constraints from both endpoints — but stored as a single compact matrix (4×4 or decomposed as 3×3 A + 3-vector b + scalar c) rather than an explicit list of planes.

The optimal position x* minimizes Q_merged(x*) via the 3×3 linear system A·x = -b (solved by SVD for robustness). The cost is Δ_merged(x*).

**Why it accumulates:** The merged quadric carries forward ALL original tangent planes transitively through all prior collapses. The cost always measures against the original fine mesh geometry, not a snapshot of the current state. No plane is ever forgotten — they're all folded into the matrix. This is the key property that makes QEM robust through aggressive simplification.

### Greedy Loop

1. Compute Q_v for all vertices from their incident face planes
2. For each edge, compute collapse cost and optimal position from Q_u + Q_v
3. Push to a priority queue (min-heap by cost)
4. Pop cheapest edge, collapse, set Q_w = Q_u + Q_v, re-push affected edges
5. Repeat until target vertex count reached

### What QEM Doesn't Do

QEM has no notion of mechanical importance. It preserves **geometric shape** (distance to tangent planes) but not **mechanical behavior** (stiffness, deformation response). A flat region and a high-curvature joint with similar tangent-plane geometry get similar priority. For simulation proxies, we need mechanically important vertices (joints, stiff conduits, creases) to persist longer — which is what the stiffness-informed methods below address.

## The Stiffness Matrix

All methods start from the thin-shell stiffness matrix **K** (3n × 3n for n vertices, 3 DOFs each):

```
K = K_membrane + K_bending
```

- **K_membrane** (CST elements): resists in-plane stretch/shear. Depends on triangle edge lengths.
- **K_bending** (discrete hinge): resists out-of-plane folding. Depends on dihedral angles.

K answers: "if I displace vertex v by δ, how much force pushes it back?" The diagonal block K_vv (3×3) is vertex v's self-stiffness; the off-diagonal block K_vj (3×3) is the coupling between v and neighbor j.

---

## Method 1: Schur Flow

**File:** `src/kms/schur.py` → `per_vertex_schur_flow`

### Idea

Measure how much stiffness "flows through" each vertex to its neighbors. Vertices with high flow are important mechanical conduits — removing them loses coupling.

### Math

For each vertex v, compute:

```
flow(v) = trace(K_vv⁻¹ · S)
```

where S is the sum of coupling products over all neighbors:

```
S = Σ_{j ∈ N(v)}  K_vj · K_jv
```

This equals the trace of the **Schur complement correction** — the total stiffness that would be redistributed to v's neighbors if v were eliminated:

```
correction = K_Bv · K_vv⁻¹ · K_vB
trace(correction) = trace(K_vv⁻¹ · K_vB · K_Bv) = trace(K_vv⁻¹ · S)
```

(The equality uses the cyclic property of trace: trace(ABC) = trace(CAB).)

### How It's Used for Simplification

The flow is a per-vertex scalar computed once from the original stiffness matrix. During greedy edge collapse:

```
collapse_cost(u, v) = min(flow[u], flow[v])
```

The vertex with lower flow is eliminated (merged into the higher-flow vertex). Position: midpoint.

### Properties

- **Fast:** O(1) per vertex (3×3 inverse + neighbor products)
- **No position dependence:** cost doesn't depend on where the merged vertex goes
- **No geometry awareness:** doesn't penalize vertex drift or surface deviation
- **Accumulation:** flow values are accumulated on collapse (flow[keep] += flow[elim])

### Limitation

Flow tells you how important a vertex is as a conduit, but not what happens geometrically when you remove it. Produces meshes with poor triangle quality and vertex drift because positioning is just midpoint.

---

## Method 2: Full Schur Mismatch Cost

**File:** `src/kms/schur.py` → `edge_cost_full`

### Idea

Compare the actual post-collapse stiffness against the ideal Schur-complement-condensed stiffness. The difference measures how much mechanical information is lost by the topology change.

### Math

When collapsing edge (u,v) → w at position p_w = (1-α)u + αv:

**Step 1: Ideal (Schur complement)**

Move u to p_w, then eliminate v's DOFs from the local stiffness:

```
K* = K_BB - K_Bv · K_vv⁻¹ · K_vB
```

This is the exact reduced stiffness if you could remove v without changing topology. No information lost.

**Step 2: Actual (reassemble on collapsed topology)**

Delete the faces shared by u and v. Remap v's surviving faces to u. Reassemble element stiffnesses on the new triangles → K'.

**Step 3: Cost**

```
cost(u, v, α) = ‖K' - K*‖²_F
```

restricted to the local patch (1-ring of u ∪ v).

### Optimal Position

Evaluate at α = 0, 0.5, 1. Fit a parabola. Pick the minimizer.

### Properties

- **Accurate:** measures actual mechanical degradation from the collapse
- **Position-dependent:** finds optimal merge position via 1D quadratic fit
- **Geometry-blind:** doesn't penalize surface deviation — mesh shrinks/drifts
- **Expensive:** requires local element reassembly per edge (~3 min for 8K edges in Python)

### Limitation

The cost only measures the topology change's effect on the stiffness operator. It doesn't penalize geometric deviation from the original surface. Meshes produced by this method collapse inward (bounding box shrinks to ~38% of original).

---

## Method 3: Stiffness Quadric (Approach 2 & 3)

**File:** `src/kms/simplify_stiffness_quadric.py`

### Idea

Encode "how much strain energy does displacing this vertex from its original position cost?" as a QEM-style quadratic form. This gives geometry awareness (penalizes drift) weighted by mechanical stiffness.

### Math: The Stiffness Quadric

For vertex v at original position x₀, with self-stiffness block K_vv:

```
E(x) = (x - x₀)ᵀ K_vv (x - x₀)
     = xᵀ K_vv x - 2xᵀ K_vv x₀ + x₀ᵀ K_vv x₀
```

This is a quadratic form with:
- A = K_vv (3×3 symmetric positive semi-definite)
- b = -K_vv x₀
- c = x₀ᵀ K_vv x₀

**What it means:** displacing vertex v from its original position costs strain energy proportional to K_vv. A stiff vertex (high K_vv eigenvalues) is expensive to move. An anisotropically stiff vertex penalizes movement more in certain directions.

### Edge Collapse Cost

On collapse of (u,v) → w:

```
Q_merged = Q_u + Q_v
cost = min_x  E_merged(x)
optimal_position = (K_uu + K_vv)⁻¹ (K_uu x₀_u + K_vv x₀_v)
```

The optimal position is the stiffness-weighted average of the two original positions — stiffer vertices pull the merge point toward themselves.

### Accumulation via Schur Complement

When v is eliminated at collapse position p, the Schur correction adds stiffness to the merged vertex. The correction is itself a proper quadric centered at the collapse position:

```
correction = K_uv · K_vv⁻¹ · K_vu        (3×3 matrix)
```

```
Q_correction(x) = (x - p)ᵀ · correction · (x - p)
                = xᵀ · correction · x  -  2xᵀ · correction · p  +  pᵀ · correction · p
```

The merged quadric is then:

```
Q_merged = Q_u + Q_v + Q_correction
```

In decomposed form:
- A_merged = A_u + A_v + correction
- b_merged = b_u + b_v + (-correction · p)
- c_merged = c_u + c_v + pᵀ · correction · p

The b and c updates are essential — they anchor the new stiffness at the collapse position. Without them (only updating A), the optimal position shifts unpredictably because A changed but the "center of resistance" didn't.

**Why this accumulates:** Each collapse adds a proper quadratic form. The merged quadric carries forward all prior stiffness contributions, just like QEM carries forward all prior tangent plane constraints. The Schur correction ensures the merged vertex becomes "stiffer" (harder to move from its current position) as it absorbs v's mechanical coupling role.

### Combined Mode (Approach 3)

Pure stiffness quadric doesn't know about the surface (it only penalizes displacement from original position, not distance to tangent planes). Combined mode adds the QEM plane quadric:

```
Q_combined = Q_plane + λ · Q_stiffness
```

- **Q_plane:** standard QEM — distance² to original tangent planes (keeps mesh on-surface)
- **Q_stiffness:** strain energy — displacement² weighted by K_vv (preserves mechanical coupling)
- **λ:** auto-calibrated so median costs match, or set manually

Both terms are (A, b, c) quadratic forms that accumulate additively. The combined quadric inherits QEM's geometry awareness and adds stiffness-informed priority.

### Properties

- **Geometry-aware** (via QEM term): stays on the original surface
- **Mechanically-informed** (via stiffness term): stiff vertices resist collapse
- **Accumulative:** both QEM and Schur correction compose additively
- **Optimal positioning:** closed-form SVD solve on 3×3 merged quadric
- **O(1) per edge** after initial K computation

---

## Method 4: Schur-Derived Skinning Weights

**File:** `src/kms/simplify_stiffness_quadric.py` → `compute_skinning_weights=True`

### Idea

The Schur complement gives the exact formula for reconstructing an eliminated vertex's displacement from its neighbors. Use this to derive physically-optimal skinning weights during simplification — no post-hoc optimization needed.

### Math: The Schur Reconstruction Formula

When vertex v is in static equilibrium (zero net force), its displacement satisfies:

```
K_vv u_v + K_vB u_B = 0
```

Solving for u_v:

```
u_v = -K_vv⁻¹ K_vB u_B = -K_vv⁻¹ Σ_j K_vj u_j
```

This says: "v's displacement is a linear combination of its neighbors' displacements, with coefficients determined by the stiffness coupling." These coefficients ARE the physically-optimal skinning weights for linear elasticity.

### Scalar Weight Extraction

The 3×3 matrix weight `-K_vv⁻¹ K_vj` gives a directional coupling. We extract a scalar weight per neighbor via Frobenius norm:

```
w_j = ‖K_vv⁻¹ K_vj‖_F / Σ_k ‖K_vv⁻¹ K_vk‖_F
```

### Propagation Through the Elimination Chain

When vertex v is eliminated, it gets Schur weights pointing to its current neighbors (some may be surviving, some may be themselves eliminated later). The skinning weight matrix W (n_fine × n_fine) records:

```
W[v, j] = w_j    (v depends on j with weight w_j)
```

After all collapses, some columns of W point to eliminated vertices. The finalization step iteratively propagates these dead columns:

```
For each eliminated column c with Schur weights {nb: w_nb}:
    For each row i where W[i, c] ≠ 0:
        Redistribute: W[i, c] → Σ_nb  W[i,c] · w_nb  into column nb
        W[i, c] = 0
Repeat until no weight remains in eliminated columns.
```

The final output W (n_fine × n_coarse) has:
- All rows summing to 1 (partition of unity)
- Nonzero entries only in surviving (coarse) vertex columns
- Soft blending across ~7 coarse vertices per fine vertex (physically-derived, not kNN)

### Using the Weights for LBS Reconstruction

```
V_fine(t) = V_fine_rest + W · (X_coarse(t) - V_coarse_rest)
```

Each fine vertex's displacement is the W-weighted blend of coarse vertex displacements. For linear deformations this is exact by construction. For large deformations it's an approximation (same limitation as all LBS).

### Properties

- **Physically optimal** for linear elasticity (exact reconstruction)
- **No training data needed** (derived from K, not from simulation frames)
- **Falls out of simplification** (no separate optimization step)
- **Smooth blending** (mean ~7 nonzeros per row, not hard Voronoi assignment)
- **Partition of unity** (all rows sum to 1)

### vs. Proxy-Asset-Gen's Approach

| | Proxy-Asset-Gen (Zheng et al.) | Ours |
|---|---|---|
| **Weights source** | Optimized from recorded simulation frames | Derived from stiffness matrix at rest |
| **Requires simulation** | Yes (chicken-and-egg: need proxy to simulate) | No |
| **Training cost** | Gradient descent, multiple epochs | Zero (analytical formula) |
| **Generalizes to unseen motion** | Only as well as training covers | Exact for linear, approximate for nonlinear |
| **Skinning model** | kNN bones + learned logits | Schur-derived sparse blending |

---

## Summary: Method Comparison

| Method | Cost metric | Position | Geometry | Stiffness | Accumulates | Skinning | Speed |
|--------|------------|----------|----------|-----------|-------------|----------|-------|
| Schur flow | trace(K_vv⁻¹ S) | Midpoint | No | Yes | Partial | No | Fast |
| Schur full | ‖K'-K*‖² | 1D fit | No | Yes | No | No | Slow |
| Stiffness quadric | xᵀ K_vv x | SVD optimal | With QEM | Yes | Yes (Schur) | Yes | Fast |
| Combined (recommended) | Q_plane + λ·Q_stiffness | SVD optimal | Yes | Yes | Yes | Yes | Fast |

**Recommended:** Combined mode (Approach 3) with `compute_skinning_weights=True`. Gets QEM-quality geometry, stiffness-informed priority, and physically-derived skinning weights in one pass.

---

## Running the Methods

```bash
# Visualize stiffness quadric cost
python scripts/stiffness_quadric_cost.py --mesh data/mesh.obj

# Compare simplification results
python scripts/stiffness_quadric_simp.py --mesh data/mesh.obj --target 128

# Run PBD simulation + LBS reconstruction evaluation
python scripts/eval_sim.py --mesh data/mesh.obj --target 128 --frames 360
```
