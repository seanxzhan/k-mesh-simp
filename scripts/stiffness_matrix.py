"""Thin-Shell Stiffness Matrix — Build and Visualize.

Builds the FEM stiffness matrix for thin shells (cloth) and visualizes
per-vertex stiffness and sparsity patterns.

Two components:
  - Membrane (CST): resists in-plane stretch and shear
  - Bending (discrete hinge): resists out-of-plane folding

Run:
  python scripts/stiffness_matrix.py                           # 20x20 grid
  python scripts/stiffness_matrix.py --mesh data/spot.obj      # custom mesh
  python scripts/stiffness_matrix.py --smoke                   # headless verification
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

from kms.mesh import TriMesh, load_obj, make_grid
from kms.laplacian import cotangent_laplacian
from kms.stiffness import membrane_stiffness_cst, bending_stiffness_hinge, shell_stiffness


def print_matrix_stats(name: str, K: sparse.csc_matrix, n_verts: int):
    print(f"\n  {name}:")
    print(f"    Size: {K.shape[0]}x{K.shape[1]} ({n_verts} verts x 3 DOFs)")
    print(f"    Nonzeros: {K.nnz} (density: {K.nnz / K.shape[0]**2:.6f})")
    print(f"    Symmetric: {np.allclose(K.toarray(), K.T.toarray(), atol=1e-12)}")
    diag = K.diagonal()
    print(f"    Diagonal range: [{diag.min():.6e}, {diag.max():.6e}]")
    print(f"    All diagonal >= 0: {bool(np.all(diag >= -1e-14))}")


def per_vertex_stiffness(K: sparse.csc_matrix, n_verts: int) -> np.ndarray:
    """Sum the diagonal stiffness entries per vertex (3 DOFs each)."""
    diag = K.diagonal()
    return diag.reshape(n_verts, 3).sum(axis=1)


def visualize_sparsity(K_total, K_membrane, K_bending, L, title_prefix: str, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, mat, title in zip(
        axes,
        [K_membrane, K_bending, K_total, L],
        ["K_membrane (CST)", "K_bending (hinge)", "K_total", "Laplacian L"],
    ):
        ax.spy(mat, markersize=0.3, color="navy")
        ax.set_title(f"{title}\nnnz={mat.nnz}", fontsize=10)
        ax.set_xlabel("DOF index")

    plt.suptitle(f"{title_prefix} — Sparsity Patterns", fontsize=12)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{title_prefix.lower().replace(' ', '_')}_sparsity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def make_cloth_grid(nx: int = 20, ny: int = 20, spacing: float = 1.0) -> TriMesh:
    xs = np.linspace(0, (nx - 1) * spacing, nx)
    ys = np.linspace(0, (ny - 1) * spacing, ny)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.zeros((nx * ny, 3))
    vertices[:, 0] = xx.ravel()
    vertices[:, 1] = yy.ravel()

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + i + 1
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + i + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return TriMesh(vertices=vertices, faces=np.array(faces, dtype=np.int64))


def run_analysis(mesh: TriMesh, label: str, out_dir: Path, smoke: bool):
    print(f"\n  Mesh: {mesh.n_verts} verts, {mesh.n_faces} faces")

    K_total, K_membrane, K_bending = shell_stiffness(mesh)
    L, M = cotangent_laplacian(mesh)

    print_matrix_stats("K_membrane (in-plane stretch/shear)", K_membrane, mesh.n_verts)
    print_matrix_stats("K_bending (out-of-plane folding)", K_bending, mesh.n_verts)
    print_matrix_stats("K_total = K_membrane + K_bending", K_total, mesh.n_verts)

    print(f"\n  Cotangent Laplacian L: {L.shape[0]}x{L.shape[1]}, nnz={L.nnz}")
    print(f"  Note: L is {L.shape[0]}x{L.shape[0]} (1 DOF/vert), "
          f"K is {K_total.shape[0]}x{K_total.shape[0]} (3 DOFs/vert)")

    print(f"\n  Sparsity comparison (nonzeros per row, average):")
    print(f"    K_membrane: {K_membrane.nnz / K_membrane.shape[0]:.1f}")
    print(f"    K_bending:  {K_bending.nnz / K_bending.shape[0]:.1f}")
    print(f"    K_total:    {K_total.nnz / K_total.shape[0]:.1f}")
    print(f"    Laplacian:  {L.nnz / L.shape[0]:.1f}")
    print(f"\n  Key insight: bending connects the 2-ring (wider sparsity than membrane)")

    if not smoke:
        visualize_sparsity(K_total, K_membrane, K_bending, L, label, out_dir)

    return mesh, K_total, K_membrane, K_bending


def smoke_check(mesh, K_total, K_membrane, K_bending, label: str):
    n = mesh.n_verts
    ndof = 3 * n

    assert K_total.shape == (ndof, ndof), f"{label}: wrong shape"
    assert K_membrane.shape == (ndof, ndof), f"{label}: wrong shape"
    assert K_bending.shape == (ndof, ndof), f"{label}: wrong shape"

    assert np.allclose(K_total.toarray(), K_total.T.toarray(), atol=1e-12), f"{label}: not symmetric"
    assert np.allclose(K_membrane.toarray(), K_membrane.T.toarray(), atol=1e-12), f"{label}: not symmetric"
    assert np.allclose(K_bending.toarray(), K_bending.T.toarray(), atol=1e-12), f"{label}: not symmetric"

    assert np.all(K_total.diagonal() >= -1e-12), f"{label}: negative diagonal"

    diff = K_total - (K_membrane + K_bending)
    assert abs(diff).max() < 1e-12, f"{label}: K_total != K_m + K_b"

    assert K_membrane.nnz > 0, f"{label}: K_membrane is empty"
    assert K_bending.nnz > 0, f"{label}: K_bending is empty"

    print(f"  SMOKE {label}: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default=None, help="Path to .obj mesh")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out_dir = Path("out")

    if args.mesh:
        print("=" * 60)
        print("Stiffness Matrix: Input Mesh")
        print("=" * 60)
        mesh = load_obj(args.mesh)
        print(f"\n  Loaded: {args.mesh}")
        label = "Input Mesh"
    else:
        print("=" * 60)
        print("Stiffness Matrix: 20x20 Cloth Grid")
        print("=" * 60)
        mesh = make_cloth_grid(20, 20, spacing=1.0)
        label = "20x20 Grid"

    mesh, K_total, K_membrane, K_bending = run_analysis(mesh, label, out_dir, args.smoke)

    if args.smoke:
        print("\n--- Smoke checks ---")
        smoke_check(mesh, K_total, K_membrane, K_bending, label)
        print("\nSMOKE: ALL PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")

        stiffness_per_vert = per_vertex_stiffness(K_total, mesh.n_verts)

        ps_mesh = ps.register_surface_mesh(
            label, mesh.vertices, mesh.faces, edge_width=0.5
        )
        ps_mesh.add_scalar_quantity(
            "total stiffness (diag sum)",
            stiffness_per_vert,
            defined_on="vertices",
            enabled=True,
            cmap="viridis",
        )

        membrane_per_vert = per_vertex_stiffness(K_membrane, mesh.n_verts)
        bending_per_vert = per_vertex_stiffness(K_bending, mesh.n_verts)

        ps_mesh.add_scalar_quantity(
            "membrane stiffness",
            membrane_per_vert,
            defined_on="vertices",
            cmap="plasma",
        )
        ps_mesh.add_scalar_quantity(
            "bending stiffness",
            bending_per_vert,
            defined_on="vertices",
            cmap="inferno",
        )

        print("\n  Polyscope window open. Showing per-vertex stiffness.")
        print("  Toggle between 'membrane stiffness' and 'bending stiffness' "
              "in the UI to see their spatial distribution.")
        ps.show()


if __name__ == "__main__":
    main()
