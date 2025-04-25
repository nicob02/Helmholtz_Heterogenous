
import torch
from core.pde import laplacian, grad
import numpy as np
import math

class ElectroThermalFunc(): 

    func_name = 'Helmholtz_Heterogenous'
    def __init__(
        self,
        eps=(4.0,2.0,1.0),
        k=(20.0,10.0,5.0),
        center=(0.5,0.5),
        r1=0.15, r2=0.30,
        sigma=0.05,
        bc_tol=1e-3
    ):
        self.eps1,self.eps2,self.eps3 = eps
        self.k1,  self.k2,  self.k3   = k
        self.cx,  self.cy            = center
        self.r1,  self.r2            = r1, r2
        self.sigma                  = sigma
        self.bc_tol                 = bc_tol

    def graph_modify(self, graph):
        """
        Build node features [x, y, eps, k, f].
        """
        x = graph.pos[:,0:1]
        y = graph.pos[:,1:2]
        dx = x - self.cx
        dy = y - self.cy
        r  = torch.sqrt(dx*dx + dy*dy)

        # 1) piecewise eps/k
        eps = torch.where(r <= self.r1,
                          self.eps1,
                   torch.where(r <= self.r2,
                               self.eps2,
                               self.eps3)).unsqueeze(1)
        k   = torch.where(r <= self.r1,
                          self.k1,
                   torch.where(r <= self.r2,
                               self.k2,
                               self.k3)).unsqueeze(1)

        # 2) Gaussian source term
        f = torch.exp(-((x-0.5)**2 + (y-0.5)**2) / (2*self.sigma**2))

        # concat into graph.x
        graph.x = torch.cat([x, y, eps, k, f], dim=-1)
        return graph

    def _ansatz_u(self, graph, u_raw):
        # no hard Dirichlet any more
        return u_raw

    def pde_residual(self, graph, u):
        """
        Now re-use graph.x[:,4] for f instead of recomputing it.
        """
        pos = graph.pos
        # extract eps,k,f from the features
        eps = graph.x[:,2:3]
        k   = graph.x[:,3:4]
        f   = graph.x[:,4:5]

        # compute grad u
        grad_u = torch.autograd.grad(
            outputs=u, inputs=pos,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]                          # [N,2]

        # flux = eps * grad_u
        flux = eps * grad_u           # [N,2]

        # divergence
        div = torch.zeros_like(u)
        for i in range(2):
            di = torch.autograd.grad(
                outputs=flux[:,i:i+1],
                inputs=pos,
                grad_outputs=torch.ones_like(flux[:,i:i+1]),
                create_graph=True
            )[0][:,i:i+1]
            div = div + di

        # residual = div(eps grad u) + k^2 u - f
        r_pde = div + (k**2)*u - f

        return r_pde, grad_u
