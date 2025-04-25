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
import numpy as np
from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad,
    sin, cos, pi
)
from ufl import conditional, le, SpatialCoordinate

def run_fem(electrode_mesh,
            coords,
            r1=0.15, r2=0.30,
            eps1=4.0, eps2=2.0, eps3=1.0,
            k1=20.0, k2=10.0, k3=5.0):
    """
    Solve ∇·(ε(x) ∇u) + k(x)^2 u = f(x) on the unit square
    with two concentric inclusions, *all* Dirichlet u=0 on ∂Ω,
    and “Helmholtz‐style” right‐hand side f(x,y).

    Returns:
      coords : (N,2) array of query points (same as graph.pos)
      VV     : length‐N array of FEM u(x,y) at those points
    """

    # 1) underlying mesh
    mesh = electrode_mesh.mesh

    # 2) P1 scalar space
    V = FunctionSpace(mesh, "CG", 1)

    # 3) trial/test
    u = TrialFunction(V)
    v = TestFunction(V)

    # 4) symbolic coords
    X     = SpatialCoordinate(mesh)
    rr2   = (X[0]-0.5)**2 + (X[1]-0.5)**2

    # 5) piecewise ε(x), k(x)
    eps = conditional(
        le(rr2, r1**2),
        Constant(eps1),
        conditional(le(rr2, r2**2),
                    Constant(eps2),
                    Constant(eps3))
    )
    kk = conditional(
        le(rr2, r1**2),
        Constant(k1),
        conditional(le(rr2, r2**2),
                    Constant(k2),
                    Constant(k3))
    )

    # 6) Helmholtz‐style forcing f(x,y)
    #    f = 2π cos(πy) sin(πx)
    #      + 2π cos(πx) sin(πy)
    #      + (x+y) sin(πx) sin(πy)
    #      − 2π² (x+y) sin(πx) sin(πy)
    f_expr = (
        2*pi * cos(pi*X[1]) * sin(pi*X[0])
      + 2*pi * cos(pi*X[0]) * sin(pi*X[1])
      + (X[0] + X[1]) * sin(pi*X[0]) * sin(pi*X[1])
      - 2*pi**2 * (X[0] + X[1]) * sin(pi*X[0]) * sin(pi*X[1])
    )

    # 7) weak form: ∫ ε ∇u·∇v + k² u v  = ∫ f v
    a = dot(eps*grad(u), grad(v))*dx + kk**2 * u*v*dx
    L = f_expr * v * dx

    # 8) Dirichlet u=0 on *all* outer boundaries
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    # 9) solve
    U = Function(V)
    solve(a == L, U, bc)

    # 10) sample at the same points as your GNN
    VV = np.empty((coords.shape[0],), dtype=np.float64)
    for i, (xi, yi) in enumerate(coords):
        VV[i] = U(Point(float(xi), float(yi)))

    return coords, VV
