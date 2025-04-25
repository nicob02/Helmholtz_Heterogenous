
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
        Build node features [x, y, eps(x), k(x), f(x,y)] all of shape [N,1].
        """
        x = graph.pos[:, 0:1]   # [N,1]
        y = graph.pos[:, 1:2]   # [N,1]
        dx = x - self.cx
        dy = y - self.cy
        r  = torch.sqrt(dx*dx + dy*dy)   # [N,1]
        
        # build constant fields of shape [N,1] automatically:
        E1 = torch.full_like(x, self.eps1)
        E2 = torch.full_like(x, self.eps2)
        E3 = torch.full_like(x, self.eps3)
        
        K1 = torch.full_like(x, self.k1)
        K2 = torch.full_like(x, self.k2)
        K3 = torch.full_like(x, self.k3)
        
        # piecewise epsilon and k
        eps = torch.where(r <= self.r1, E1,
                  torch.where(r <= self.r2, E2, E3))
        k   = torch.where(r <= self.r1, K1,
                  torch.where(r <= self.r2, K2, K3))
        
        # f(x,y) = 2π cos(πy) sin(πx)
        #         + 2π cos(πx) sin(πy)
        #         + (x+y) sin(πx) sin(πy)
        #         - 2π² (x+y) sin(πx) sin(πy)
        f = (
            2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
            + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
            + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
            - 2 * (math.pi ** 2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        )

        # first embed just the spatial coords:
        coords = torch.cat([x,y], dim=-1)
        fourier = self.ff(coords)   # __init__: self.ff = FourierFeatures(2, mapping_size=64, scale=20)
        graph.x = torch.cat([fourier, eps, k, f], dim=-1)

        return graph


    def _ansatz_u(self, graph, u_raw):
        
        x, y = graph.pos[:,0:1], graph.pos[:,1:2]
        # Distance factor D(x,y) that vanishes exactly on ALL four edges:
        D = ( torch.tanh(math.pi * x) 
            * torch.tanh(math.pi * (1.0 - x))
            * torch.tanh(math.pi * y)
            * torch.tanh(math.pi * (1.0 - y)) )
        # G(x,y) = 0 everywhere on boundary:
        G = torch.zeros_like(x)
        return G + D * u_raw

    def pde_residual(self, graph, u):
        """
        Compute
           r_pde = div( eps * grad u ) + k^2 u - f
        at each node, returning (r_pde, grad_u).
        """
        pos = graph.pos
        eps = graph.x[:, 2:3]    # [N,1]
        k   = graph.x[:, 3:4]    # [N,1]
        f   = graph.x[:, 4:5]    # [N,1]

        # 1) ∇u  → grad_u [N,2]
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=pos,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # 2) flux = ε ∇u
        flux = eps * grad_u       # [N,2]

        # 3) divergence of flux
        div = torch.zeros_like(u)
        for d in range(2):
            div_d = torch.autograd.grad(
                outputs=flux[:, d:d+1],
                inputs=pos,
                grad_outputs=torch.ones_like(flux[:, d:d+1]),
                create_graph=True
            )[0][:, d:d+1]
            div = div + div_d

        # 4) PDE residual
        r_pde = div + (k**2)*u - f

        return r_pde, grad_u
