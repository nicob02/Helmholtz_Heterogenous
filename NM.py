from petsc4py import PETSc
import numpy as np
import sys
import matplotlib.pyplot as plt
from core.geometry import ElectrodeMesh
from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad
)
from ufl import conditional, le, SpatialCoordinate, exp
import numpy as np

def run_fem(electrode_mesh, coords=None, r1=0.15, r2=0.30, eps1=4.0, eps2=2.0, eps3=1.0, k1=20.0, k2=10.0, k3=5.0):
    """
    Solve ∇·(ε∇u) + k^2 u = f  on the unit square with
    two concentric inclusions, *all*-Neumann boundary,
    and Gaussian source f.
    """
    mesh = electrode_mesh.mesh
    V_space = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V_space)
    v = TestFunction(V_space)

    X = SpatialCoordinate(mesh)
    r2sym = (X[0]-0.5)**2 + (X[1]-0.5)**2

    # piecewise eps/k
    eps = conditional(
        le(r2sym, r1**2), Constant(eps1),
        conditional(le(r2sym, r2**2), Constant(eps2), Constant(eps3))
    )
    kk = conditional(
        le(r2sym, r1**2), Constant(k1),
        conditional(le(r2sym, r2**2), Constant(k2), Constant(k3))
    )

    # Gaussian source
    f_expr = exp(-r2sym/(2*sigma**2))

    # weak form
    a = dot(eps*grad(u), grad(v))*dx + kk**2 * u*v*dx
    L = f_expr*v*dx

    # solve with pure Neumann: no DirichletBC at all
    U = Function(V_space)
    solve(a == L, U)

    # sample at GNN nodes
    VV = np.empty((coords.shape[0],), dtype=np.float64)
    for i,(xi,yi) in enumerate(coords):
        VV[i] = U(Point(float(xi),float(yi)))

    return coords, VV
