import torch
from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import compute_steady_error, render_results
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from functions import ElectroThermalFunc as Func
import os
import matplotlib.pyplot as plt
from NM import run_fem 
from torch_geometric.data import Data



out_ndim = 1

dens=65
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name    #FIGURE THIS OUT
device = torch.device(0)

func_main = Func(eps=(4.0,2.0,1.0),
    k  =(20.0,10.0, 5.0),
    center=(0.5,0.5),
    r1=0.15,
    r2=0.30,
    bc_tol=1e-2
)


mesh = ElectrodeMesh(ru=(1, 1), lb=(0, 0), density=65)
graph = mesh.getGraphData()
mapping_size = 64
node_input_size = 2*mapping_size + 3  # = 131
edge_input_size = 3                    # however many edge features you have

raw_node_input_size = 5  

model = msgPassing(
    message_passing_num=3,
    node_input_size=raw_node_input_size,   # <<— 5, not 131
    edge_input_size=3,
    ndim=out_ndim,
    device=device,
    fourier_mapping_size=64,
    fourier_scale=0.01,
    model_dir=ckptpath
)
model.load_model(ckptpath)
model.to(device)
model.eval()
test_steps = 20

test_config = parse_config()

#model = kwargs['model'] # Extracts the model's dictioanry with the weights and biases values
setattr(test_config, 'device', device)   
setattr(test_config, 'model', model)
setattr(test_config, 'test_steps', test_steps)
setattr(test_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref)
setattr(test_config, 'ndim', out_ndim)
setattr(test_config, 'graph_modify', func_main.graph_modify)
setattr(test_config, 'graph', graph)
setattr(test_config, 'density', dens)
setattr(test_config, 'func_main', func_main)
setattr(test_config, 'lambda_neu', 1.0) 
setattr(test_config, 'lambda_if', 1.0)      

#-----------------------------------------

print('************* model test starts! ***********************')
predicted_results = modelTester(test_config)

pos_np = graph.pos.cpu().numpy()
x, y   = pos_np[:,0], pos_np[:,1]

coords_fem, V_vals_fem = run_fem(electrode_mesh=mesh, coords=graph.pos.cpu().numpy()) 

# 3) Compute and print relative L2 errors
err_V = compute_steady_error(predicted_results, V_vals_fem, test_config)
print(f"Rel L2 error Voltage:     {err_V:.3e}")

render_results(predicted_results, V_vals_fem, graph, filename="ForcingFunction.png")



fig, axes = plt.subplots(1, 2, figsize=(12,5), tight_layout=True)
# Hz
sc0 = axes[0].scatter(x, y, c=predicted_results.flatten(), cmap='viridis', s=5)
axes[0].set_title("Predicted Voltage Distribution")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
plt.colorbar(sc0, ax=axes[0], shrink=0.7)

plt.savefig("Helmholz_Heterogenous.png", dpi=300)
plt.close(fig)
print("Done — predictions plotted to Helmholz_Heterogenous.png")

"""
u_exact = func_main.exact_solution(graph)  
u_exact_np  = u_exact.detach().cpu().numpy()
# 2) Compute exact & error
rel_l2 = compute_steady_error(predicted_results, u_exact_np, test_config)
print(f"Relative L2 error: {rel_l2:.3e}")

# 3) Render the three‐panel result
render_results(predicted_results, u_exact_np, graph, filename="helmholtz_steady.png")
"""
