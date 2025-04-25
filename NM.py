from petsc4py import PETSc
import numpy as np
import sys
import matplotlib.pyplot as plt
from core.geometry import ElectrodeMesh
from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad, near
)
from ufl import conditional, le, SpatialCoordinate, exp
import numpy as np
def run_fem(electrode_mesh,
            coords,
            r1=0.15, r2=0.30,
            eps1=4.0, eps2=2.0, eps3=1.0,
            k1=20.0, k2=10.0, k3=5.0,
            sigma=0.1):
    """
    Solve  ∇·(ε(x) ∇u) + k(x)^2 u = f(x)
    on the unit square with two concentric inclusions and
    a Gaussian source f, *all*-Neumann outer edges,
    plus a single Dirichlet pin at the center to kill the constant nullspace.
    Returns
      coords : (N,2) array of query points (same as graph.pos)
      VV     : length-N array of FEM solution u(x,y) at those points
    """

    # 1) Extract the underlying DOLFIN mesh
    mesh = electrode_mesh.mesh

    # 2) P1 (CG1) scalar function space
    V = FunctionSpace(mesh, "CG", 1)

    # 3) Trial & test
    u = TrialFunction(V)
    v = TestFunction(V)

    # 4) Symbolic coords and radius²
    X      = SpatialCoordinate(mesh)
    r2sym  = (X[0] - 0.5)**2 + (X[1] - 0.5)**2

    # 5) Piecewise ε(x) and k(x)
    eps = conditional(
        le(r2sym, r1**2),
        Constant(eps1),
        conditional(
            le(r2sym, r2**2),
            Constant(eps2),
            Constant(eps3)
        )
    )
    kk = conditional(
        le(r2sym, r1**2),
        Constant(k1),
        conditional(
            le(r2sym, r2**2),
            Constant(k2),
            Constant(k3)
        )
    )

    # 6) Gaussian source f(x)
    f_expr = exp(-r2sym/(2.0*sigma**2))

    # 7) Weak form: ∫ ε ∇u⋅∇v dx + ∫ k² u v dx = ∫ f v dx
    a = dot(eps*grad(u), grad(v))*dx + kk**2 * u*v*dx
    L = f_expr * v * dx

    # 8) Pure Neumann → no boundary terms in a/L
    #    But we must kill the constant nullspace:
    #    Pin u(0.5,0.5)=0 as a single Dirichlet dof
    def at_center(x, on_bnd):
        return near(x[0], 0.5) and near(x[1], 0.5)

    bc_center = DirichletBC(V, Constant(0.0), at_center, method="pointwise")

    # 9) Solve
    U = Function(V)
    solve(a == L, U, bc_center)

    # 10) Sample at the requested coords
    VV = np.empty((coords.shape[0],), dtype=np.float64)
    for i, (xi, yi) in enumerate(coords):
        VV[i] = U(Point(float(xi), float(yi)))

    return coords, VV
