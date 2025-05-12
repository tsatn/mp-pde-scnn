import math
import torch
import torch.nn as nn
from torch import matmul
import torch.nn.functional as F
from pathlib import Path
import networkx as nx
from torch_geometric.data import Data
import numpy as np
from common.simplicial_utils import build_complex_from_edge_index 
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
# from third_party.snn import chebyshev
# from third_party.snn.scnn.chebyshev import normalize 
from common.simplicial_utils import normalize

# old version    
class SimplicialConvolution(nn.Module):
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 B: torch.Tensor | None = None,   # ← now optional
                 dim: int = 0,                    # PyG keyword
                 **kwargs):
        super().__init__()
        self.B   = B            # can be None
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x_src: torch.Tensor) -> torch.Tensor:
        x_proj = self.lin(x_src)                   # (|src|, C_out)

        # If boundary operator is missing, just return the projection
        if self.B is None:
            return x_proj

        # Otherwise perform the simplicial message‑passing
        return torch.sparse.mm(self.B, x_proj)
    
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# class SimplicialProcessor(nn.Module):
#     def __init__(self, hidden_dim, boundary_maps, alpha=0.5):
#         super().__init__()
#         self.boundary_maps = boundary_maps
        
#         # Simplicial convolutions clearly defined for nodes, edges, triangles
#         self.conv0 = SimplicialConvolution(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim,
#             dim=0,
#             laplacian_type="normalized"
#         )
#         self.conv1 = SimplicialConvolution(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim,
#             dim=1,
#             laplacian_type="normalized"
#         )
#         self.conv2 = SimplicialConvolution(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim,
#             dim=2,
#             laplacian_type="normalized"
#         )

#         # Parameter to balance between simplicial interactions
#         self.alpha = nn.Parameter(torch.tensor(alpha))
#         self.swish = Swish()

#     def forward(self, X0, X1, X2):    
#         X0_lower = self.conv0(X0, self.boundary_maps['B1'])
#         X0_upper = matmul(self.boundary_maps['B2'], X2)
#         X0_out = self.swish(self.alpha * X0_lower + (1 - self.alpha) * X0_upper)
        
#         X1_lower = self.conv1(X1, self.boundary_maps['B1'])
#         X1_upper = matmul(self.boundary_maps['B2'].t(), X2)
#         X1_out = self.swish(0.5 * (X1_lower + X1_upper))
        
#         X2_out = self.swish(self.conv2(X2, self.boundary_maps['B2']))
#         return X0_out, X1_out, X2_out

class SimplicialProcessor(nn.Module):
    def __init__(self, hidden_dim, boundary_maps, alpha=0.5):
        super().__init__()
        self.boundary_maps = boundary_maps
        
        self.conv0 = SimplicialConvolution(hidden_dim, hidden_dim, dim=0, laplacian_type="normalized")
        self.conv1 = SimplicialConvolution(hidden_dim, hidden_dim, dim=1, laplacian_type="normalized")
        self.conv2 = SimplicialConvolution(hidden_dim, hidden_dim, dim=2, laplacian_type="normalized")

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.swish = Swish()

    def forward(self, X0, X1, X2):    
        X0_lower = self.conv0(X0, self.boundary_maps['B1'])
        X0_upper = matmul(self.boundary_maps['B2'], X2)
        X0_out = self.swish(self.alpha * X0_lower + (1 - self.alpha) * X0_upper)
        
        X1_lower = self.conv1(X1, self.boundary_maps['B1'])
        X1_upper = matmul(self.boundary_maps['B2'].t(), X2)
        X1_out = self.swish(0.5 * (X1_lower + X1_upper))
        
        X2_out = self.swish(self.conv2(X2, self.boundary_maps['B2']))
        return X0_out, X1_out, X2_out


class SCNPDEModel(nn.Module):
    def __init__(self, mesh, time_steps, feat_dims, hidden=128):
        super().__init__()
        self.is_graph_model = True   
        self.expects_graph = True  
        
        # Store boundary matrices from mesh
        self.boundary_maps = {
            'B1': mesh.B1,
            'B2': mesh.B2
        }

        # Encoders (same as MP-PDE)
        self.enc0 = nn.Sequential(
            nn.Linear(feat_dims['node'], hidden),
            Swish()
        )
        self.enc1 = nn.Sequential(
            nn.Linear(feat_dims['edge'], hidden),
            Swish()
        ) if 'edge' in feat_dims else None
        self.enc2 = nn.Sequential(
            nn.Linear(feat_dims['triangle'], hidden),
            Swish()
        ) if 'triangle' in feat_dims else None
        
        # SCNN Processor
        self.processor = SimplicialProcessor(hidden, self.boundary_maps)
        
        # Temporal bundling (same as MP-PDE)
        self.temporal_steps = 3
        self.temporal_proj = nn.Linear(hidden * self.temporal_steps, hidden)
        
        # Decoder (same as MP-PDE)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, feat_dims['node'], kernel_size=3, padding=1),
            Swish()
        )
    
    def forward(self, data: Data):
        """
        Public entry‑point: receives a PyG graph, extracts feature
        matrices and delegates to the private _forward().
        """
        # node‑features = past u‑values  +  (t,x) coordinates
        X0 = torch.cat([data.x, data.pos], dim=-1)        # [N, tw+2]
        
        # edge / triangle features may be missing → keep None
        X1 = getattr(data, 'edge_attr', None)
        X2 = getattr(data, 'tri_attr',  None)
        
        return self._forward(X0, X1, X2)
    
    
    def _forward(self, X0, X1=None, X2=None):
        # Encode features
        X0h = self.enc0(X0)
        X1h = self.enc1(X1) if (self.enc1 and X1 is not None) else torch.zeros_like(X0h)
        X2h = self.enc2(X2) if (self.enc2 and X2 is not None) else torch.zeros_like(X0h)
        # Temporal bundling
        bundled = [X0h]
        for _ in range(self.temporal_steps-1):
            X0h, X1h, X2h = self.processor(X0h, X1h, X2h)
            bundled.append(X0h)
            
        # Combine temporal steps
        temporal_features = self.temporal_proj(torch.cat(bundled, dim=-1))
        
        # Decode to physical space
        return self.decoder(temporal_features.unsqueeze(-1)).squeeze(-1)


def build_scn_maps(mesh, max_order: int = 2):
    """
    Build boundary / incidence maps for the given PyG mesh.

    Parameters
    ----------
    mesh : torch_geometric.data.Data
        Must contain   • edge_index  (2,E)   • num_nodes
    max_order : int
        1 → nodes+edges only, 2 → also triangles.

    Returns
    -------
    dict  with sparse tensors
        • 'A01' : |V|×|E|   (edge  → node)             always
        • 'A02' : |V|×|T|   (triangle → node)          if max_order≥2
        • 'A12' : |E|×|T|   (triangle → edge)          if max_order≥2
        • 'triangles' : LongTensor [n_tri,3]           if max_order≥2
    """
    comp = build_complex_from_edge_index(mesh.edge_index, max_order=max_order)

    maps = {
        'A01': comp['A01']
    }
    if max_order >= 2:
        maps.update({
            'A02': comp['A02'],
            'A12': comp['A12'],
            'triangles': comp['triangles'],
        })

    return maps

# def normalize_boundary(B):
#     row_sum = torch.sparse.sum(B, dim=1).to_dense()
#     col_sum = torch.sparse.sum(B, dim=0).to_dense()

#     row_norm = torch.from_numpy(1 / np.sqrt(row_sum + 1e-8))
#     col_norm = torch.from_numpy(1 / np.sqrt(col_sum + 1e-8))
    
#     indices = torch.from_numpy(np.vstack([B.row, B.col]))
#     values = torch.from_numpy(B.data) * row_norm[B.row] * col_norm[B.col]
#     return torch.sparse_coo_tensor(indices, values, B.shape)

def normalize_boundary(B: torch.Tensor):
    row_sum = torch.sparse.sum(B, dim=1).to_dense()
    col_sum = torch.sparse.sum(B, dim=0).to_dense()

    row_norm = 1 / torch.sqrt(row_sum + 1e-8)
    col_norm = 1 / torch.sqrt(col_sum + 1e-8)

    indices = B._indices()
    values = B._values() * row_norm[indices[0]] * col_norm[indices[1]]

    return torch.sparse_coo_tensor(indices, values, B.shape)
