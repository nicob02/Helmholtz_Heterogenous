from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from core.utils.gnnutils import copy_geometric_data
import os

import torch
import torch.nn as nn
from torch_geometric.data import Data
from .model import EncoderProcesserDecoder
from .utils.fourier_features import FourierFeatures

class Simulator(nn.Module):

    def __init__(self,
                 message_passing_num: int,
                 node_input_size: int,
                 edge_input_size: int,
                 ndim: int,
                 device,
                 fourier_mapping_size: int = 64,
                 fourier_scale: float = 10.0,
                 model_dir: str = 'checkpoint/simulator.pth',
                ) -> None:
        super().__init__()

        # 1) build Fourier front‐end
        self.ff = FourierFeatures(
            in_dim=2,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale
        ).to(device)

        # 2) adjust node_input_size:
        #    we remove the two raw [x,y] dims and replace them by 2*m Fourier dims
        ff_dim = 2 * fourier_mapping_size
        other_node_dims = node_input_size - 2
        new_node_input_size = ff_dim + other_node_dims

        # 3) instantiate your GNN
        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=new_node_input_size,
            edge_input_size=edge_input_size,
            ndim=ndim
        ).to(device)

        self.model_dir = model_dir
        self.device    = device

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.bias, b=0.001)

    def forward(self, graph: Data, **argv):
        # graph.pos: [N,2]
        pos = graph.pos.to(self.device)

        # 1) compute Fourier features from the raw positions
        fourier_feats = self.ff(pos)                # [N, 2*m]

        # 2) grab the rest of the node features (everything except raw x,y)
        #    we assume graph.x was built earlier to be [x,y,...other physics dims...]
        #    so we drop [:,:2] and keep [:,2:]
        other_feats = graph.x.to(self.device)[:, 2:]  # [N, other_node_dims]

        # 3) splice them together
        graph.x = torch.cat([fourier_feats, other_feats], dim=-1)

        # 4) run through your GNN
        predicted = self.model(graph)  
        predicted.requires_grad_() 
        return predicted
    
    def save_model(self, optimizer=None):
        path = os.path.dirname(self.model_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        optimizer_dict = {}
        optimizer_dict.update({'optimizer': optimizer.state_dict()})    # Learning rate/optimization params
            
        to_save_dict ={'model':self.state_dict()}   # Model's weight params
        to_save_dict.update(optimizer_dict)
        
        torch.save(to_save_dict, self.model_dir)
        
    def load_model(self, model_dir=None, optimizer=None):

        if model_dir is None:
            model_dir = self.model_dir
        
        tmp = torch.load(model_dir, map_location='cpu')
        # print(tmp)
        dicts = tmp['model']
        self.load_state_dict(dicts, strict=True)
        
        if optimizer is None: return        
        optimizer.load_state_dict(tmp['optimizer'])
        

            
