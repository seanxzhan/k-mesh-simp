# Discussion Log: Physics-Informed Mesh Simplification

## Context

Exploring a simulation-aware mesh simplification metric for the proxy-asset-generation pipeline (Zheng et al. 2024). The goal: simplify a visual mesh into a coarse proxy that, when simulated with PBD and transferred back via LBS, produces realistic fine-mesh deformation.

---

## What does the spectral mesh simplification project use instead of QEM?

**Spectral commutativity cost** — measures how much collapsing an edge disrupts the commutativity between the Laplace-Beltrami operator and the restriction (fine→coarse) map, evaluated on the first K eigenvectors:

```
E = ||PM⁻¹LF − M̃⁻¹L̃PF||²_M̃
```

**Why not QEM?** QEM preserves visual appearance (distance to tangent planes) but destroys intrinsic spectral properties. Any downstream geometry processing that depends on the Laplacian (functional maps, shape matching, HKS/WKS) produces poor results on QEM-simplified meshes.

---

## What would "simulation-aware" mean?

Three levels of increasing awareness:

| Level | What the metric sees | Example |
|---|---|---|
| Geometry-only | Vertex positions, normals, tangent planes | QEM |
| Reconstruction-aware | How well the coarse mesh reproduces the visual mesh via LBS at pre-recorded frames | Zheng's joint ablation |
| Simulation-aware | How collapsing this edge changes the coarse mesh's dynamic response to forces | Proposed metric |

The key differentiator: a simulation-aware metric penalizes collapses that produce bad mechanical properties (non-uniform stiffness, poor constraint propagation, unstable time-stepping) even if the resulting mesh statically fits the visual mesh well.

---

## Is a simulation-aware metric possible at the same resolution?

Yes — two meshes at the same vertex count can have wildly different simulation behavior. The proxy-asset paper's own ablation proves this: their joint-optimization baseline produces 128-vertex meshes that geometrically fit better but simulate terribly because non-uniform triangles break PBD stiffness.

---

## The analogy between spectral and simulation-aware

"Does collapsing this edge disrupt the coarse mesh's ability to reproduce the fine mesh's natural vibration modes?" (spectral)

vs.

"Does collapsing this edge disrupt the coarse mesh's ability to reproduce the fine mesh's natural deformation patterns?" (simulation-aware)

| | Spectral | Simulation-aware |
|---|---|---|
| Operator | Cotangent Laplacian L | FEM stiffness matrix K |
| Modes preserved | Geometric vibration eigenvectors | Elastic deformation eigenvectors |
| "Disruption" means | Laplacian eigenvalues/vectors shift | Natural deformation patterns lost |
| Downstream use | Geometry processing (fmaps, HKS) | Physically plausible cloth motion |

---

## Sensitivity-based approach (adjoint method)

Given a deformation error:
```
E_deformation = Σ_t Σ_v || u_fine(v, t) − û(v, t) ||²
```

The adjoint method gives ∂E/∂(mesh parameters) in one backward pass — cost is roughly 1× the forward simulation regardless of parameter count:
- One forward sim → get û and all intermediate states
- One backward pass → get sensitivity of E to every vertex position / spring constant / mass
- Per edge: estimate collapse cost from the local sensitivity

---

## Proxy-asset paper's joint optimization ablation

Their ablation is essentially **QEM lifted through the skinning map** — "if I move proxy vertex j to the collapse point, how much does each skinned visual vertex deviate from its tangent planes?" It's a reconstruction loss (per-frame static geometry fitting), not simulation-aware. It failed because optimizing for static reconstruction produces meshes that look correct when driven by pre-recorded motion but simulate incorrectly when you run PBD on them (non-uniform triangle problem).

---

## QEM's key insight: error accumulation

When you collapse edge (A, B) → C, QEM sums quadric matrices: Q_C = Q_A + Q_B. The new quadric encodes all original planes that A and B were responsible for — transitively through all prior collapses. The cost v^T Q v always measures against the original fine mesh, implicitly.

**Why spectral can't accumulate:** The spectral cost depends on the current mesh's Laplacian (L̃, M̃), which is recomputed from current triangle geometry after each collapse. It's a snapshot comparison, not an accumulated error. You can't fold it into a per-vertex matrix that carries forward.

---

## Schur complement: the mechanical analog of QEM accumulation

When you eliminate a DOF from a stiffness matrix:
```
K_reduced = K_BB - K_BA * K_AA⁻¹ * K_AB
```

This folds the eliminated vertex's mechanical influence into the remaining DOFs — exactly like QEM folds geometric planes. The blocks:
- K_AA — self-stiffness of the eliminated vertex
- K_AB / K_BA — coupling between eliminated vertex and its neighbors  
- K_BB — stiffness among kept vertices

The correction term K_BA * K_AA⁻¹ * K_AB says "the stiffness that used to flow through A between its neighbors now gets folded directly into B-to-B connections."

**Properties:**
- Exact for linear elasticity (no approximation, no time-stepping needed)
- Local to compute (only touches the eliminated vertex's rows/columns)
- Composable (eliminate vertices in any order, result is the same reduced system)
- Accumulative (each elimination folds more of the original system into the remaining DOFs)

---

## Does this require volumetric meshes?

No. For cloth/shells, the stiffness matrix is defined directly on the triangle mesh:
- Membrane stiffness: CST per triangle (in-plane stretch/shear)
- Bending stiffness: discrete hinge per interior edge (out-of-plane folding)

Both are surface-only operators. Same sparsity pattern as the Laplacian (1-ring for membrane, 2-ring for bending).

---

## Optimal vertex positioning

Unlike QEM (where error is exactly quadratic in vertex position → solve Av = -b), the compliance-based cost is not quadratic in vertex position. The stiffness entries depend on triangle geometry (rational functions), and the Schur complement involves K_AA⁻¹ (nonlinear).

Practical options:
- 1D quadratic fit along the edge (sample α ∈ {0, 0.5, 1}, fit parabola) — same as spectral paper
- Linearize K w.r.t. position (Taylor expand)
- Just use midpoint

---

## The stiffness matrix

The stiffness matrix K answers: "if I displace vertex v by δ, how much force pushes it back?"

```
f = K u     (3n × 3n system, 3 DOFs per vertex)
```

### Membrane (CST — Constant Strain Triangle)

Per triangle: set up local 2D frame → compute strain-displacement matrix B → element stiffness Ke = area × thickness × Bᵀ D B → rotate to 3D → scatter into global K.

The formula comes from minimizing elastic strain energy U = (1/2) ∫ εᵀDε dV. Since ε = Bu (constant over the triangle) and volume = area × thickness, the integral collapses.

Material model: linear elasticity (Hooke's law), plane stress assumption. NOT Neo-Hookean — linear means K is constant (computed once), valid for small strains. For the simplification metric this is the right choice because we need a fixed K that accumulates cleanly.

### Bending (discrete hinge)

Per interior edge: compute dihedral angle gradient w.r.t. 4 hinge vertices → element stiffness is rank-1 outer product Ke = coeff × grad⊗grad.

Comes from discrete differential geometry (Grinspun et al. "Discrete Shells" 2003), not standard FEM. Curvature is concentrated at edges; the energy is a function of dihedral angles.

### Comparison with cotangent Laplacian

| Property | Laplacian L | Stiffness K |
|----------|------------|-------------|
| Size | n × n (1 DOF/vert) | 3n × 3n (3 DOFs/vert) |
| Connectivity | 1-ring | 1-ring (membrane) + 2-ring (bending) |
| What it encodes | Geometric smoothness | Mechanical resistance |
| Null space | Constants (1D) | Rigid body motions (6D) |

The cotangent Laplacian is the membrane stiffness of a 1D scalar field on the surface — same DNA.

---

## Does stiffness-based simplification make sense for PBD simulation?

The mismatch matters less than expected. PBD's stretch constraints are proportional to edge rest lengths, and bending constraints to rest dihedral angles — both geometric quantities that K captures. A region that's stiff in K will also be stiff in PBD. The stiffness metric implicitly enforces what PBD needs (uniformity, good triangle quality) without running PBD.

Worth testing: simplify with stiffness-based cost → simulate with PBD → compare deformation error against QEM and Voronoi at the same vertex count.

---

## Can stiffness matrices be constructed on any connected triangle mesh?

Yes. Only need:
- Triangles (for membrane via CST)
- Interior edges shared by two triangles (for bending via discrete hinge)

Boundary edges get no bending stiffness (physically correct — free edges bend freely). No volumetric mesh or special topology required.

---

## Do people use linear elasticity for cloth simulation?

Not for final production:
- **Games (PBD/XPBD):** No constitutive model — springs + position constraints
- **Film VFX:** Nonlinear models (StVK, Baraff-Witkin strain limiting)
- **Research:** Neo-Hookean to anisotropic woven fabric models

Where linear elasticity still shows up:
- Tangent stiffness at each Newton step of implicit integration
- Modal analysis / eigenanalysis around rest pose
- Pre-computation passes needing a fixed K (our Schur complement idea)

For the simplification metric: linear K captures "which edges are mechanically important" correctly, even if it won't reproduce the full nonlinear trajectory.
