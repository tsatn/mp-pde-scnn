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
import scipy.sparse as sp
from common.simplicial_utils import Coboundary

class SimplicialConvolution(nn.Module):
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 B: torch.Tensor | None = None,
                 dim: int = 0,
                 **kwargs):
        super().__init__()
        self.B   = B
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.dim = dim

    def forward(self, x_src: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        x_proj = self.lin(x_src)
        # For node update (dim=0), just return x_proj (no boundary op)
        if self.dim == 0:
            return x_proj
        # For edge/triangle update, apply boundary operator if provided
        B = B if B is not None else self.B
        if B is None:
            return x_proj
        return torch.sparse.mm(B, x_proj)
    
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class SimplicialProcessor(nn.Module):
    def __init__(self, hidden_dim, boundary_maps, alpha=0.5):
        super().__init__()
        self.boundary_maps = {
            'B1': boundary_maps['B1'].to(torch.float32),
            'B2': boundary_maps['B2'].to(torch.float32) if boundary_maps['B2'] is not None else None
        }
        
        # Initialize convolutions
        self.conv0 = SimplicialConvolution(hidden_dim, hidden_dim, dim=0)
        self.conv1 = SimplicialConvolution(hidden_dim, hidden_dim, dim=1)
        self.conv2 = SimplicialConvolution(hidden_dim, hidden_dim, dim=2)

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.swish = Swish()

    def forward(self, X0, X1, X2):
        """
        Input shapes:
        X0: [B, hidden, N] - node features
        X1: [B, hidden, E] - edge features (or None)
        X2: [B, hidden, T] - triangle features (or None)
        """
        # Ensure inputs are float32
        X0 = X0.to(torch.float32)
        if X1 is not None:
            X1 = X1.to(torch.float32)
        if X2 is not None:
            X2 = X2.to(torch.float32)

        B = X0.size(0)
        hidden = X0.size(1)

        # Node update (dim=0)
        X0_flat = X0.transpose(1, 2).reshape(-1, hidden)  # [B*N, hidden]
        X0_lower = self.conv0(X0_flat)  # [B*N, hidden]
        X0_lower = X0_lower.view(B, -1, hidden).transpose(1, 2)  # [B, hidden, N]

        # Apply triangle influence if exists
        if X2 is not None:
            # [B, hidden, T] x [T, N] -> [B, hidden, N]
            X0_upper = torch.matmul(X2, self.boundary_maps['B2'].to_dense())
            X0_out = self.swish(self.alpha * X0_lower + (1 - self.alpha) * X0_upper)
        else:
            X0_out = self.swish(X0_lower)

        # Edge update (dim=1)
        X1_out = None
        if X1 is not None:
            X1_flat = X1.transpose(1, 2).reshape(-1, hidden)  # [B*E, hidden]
            X1_lower = self.conv1(X1_flat)  # [B*E, hidden]
            num_edges = X1.size(2)
            X1_lower = X1_lower.view(B, num_edges, hidden).transpose(1, 2)  # [B, hidden, E]
            
            if X2 is not None:
                # [B, hidden, T] x [T, E] -> [B, hidden, E]
                X1_upper = torch.matmul(X2, self.boundary_maps['B2'].t().to_dense())
                X1_out = self.swish(0.5 * (X1_lower + X1_upper))
            else:
                X1_out = self.swish(X1_lower)

        # Triangle update (dim=2)
        X2_out = None
        if X2 is not None:
            X2_flat = X2.transpose(1, 2).reshape(-1, hidden)  # [B*T, hidden]
            X2_out = self.conv2(X2_flat)  # [B*T, hidden]
            num_triangles = X2.size(2)
            X2_out = X2_out.view(B, num_triangles, hidden).transpose(1, 2)  # [B, hidden, T]
            X2_out = self.swish(X2_out)

        return X0_out, X1_out, X2_out

class SCNPDEModel(nn.Module):
    def __init__(self, mesh, time_steps, feat_dims, hidden=128):
        super().__init__()
        self.is_graph_model = True   
        self.expects_graph = True  
        self.mesh = mesh
        self.temporal_steps = 3
        self.hidden = hidden
        
        # Convert boundary matrices to float32
        self.boundary_maps = {
            'B1': mesh.B1.to(torch.float32),
            'B2': mesh.B2.to(torch.float32) if hasattr(mesh, 'B2') else None
        }
        
        # Input feature dimensions should match
        self.enc0 = nn.Sequential(
            nn.Linear(feat_dims['node'], hidden, dtype=torch.float32),  # node features → hidden
            Swish()
        )
        
        # Coboundary operators
        self.edge_coboundary = Coboundary(C_in=hidden, C_out=hidden) 
        self.tri_coboundary = Coboundary(C_in=hidden, C_out=hidden)
        
        # Edge and triangle encoders
        self.enc1 = nn.Sequential(
            nn.Linear(hidden, hidden, dtype=torch.float32),
            Swish()
        ) if 'edge' in feat_dims else None
        
        self.enc2 = nn.Sequential(
            nn.Linear(hidden, hidden, dtype=torch.float32),
        ) if 'triangle' in feat_dims else None
        
        # SCNN Processor
        self.processor = SimplicialProcessor(hidden, self.boundary_maps)
        
        # Temporal bundling
        self.temporal_steps = 3
        self.temporal_proj = nn.Linear(hidden * self.temporal_steps, hidden, dtype=torch.float32)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, time_steps, kernel_size=1, dtype=torch.float32),
            Swish()
        )
        
        # Add temporal processing
        self.temporal_processor = TemporalSimplicialProcessor(hidden, self.boundary_maps)
        
        # Add stable convolutions
        self.node_conv = StableSimplicialConvolution(hidden)
        self.edge_conv = StableSimplicialConvolution(hidden)
        
        # Add physics-informed component
        self.physics_processor = PhysicsInformedProcessor(hidden, self.boundary_maps)
        
    def forward(self, data: Data):
        # Ensure input tensors are float32
        X0 = torch.cat([
            data.x.to(torch.float32), 
            data.pos.to(torch.float32)
        ], dim=-1)
        
        # Node features + coordinates
        X0 = torch.cat([data.x, data.pos], dim=-1)  # shape [B*N, tw+2]
        X0 = X0.float()
        
        # Get batch size and number of nodes
        B = data.batch.max().item() + 1
        N = self.mesh.num_nodes
        
        # Reshape for encoder
        X0 = X0.view(B*N, -1)  # [B*N, C_in]
        
        # Encode node features
        X0h = self.enc0(X0)        # [B*N, hidden]
        X0h = X0h.view(B, N, -1)   # [B, N, hidden]
        X0h = X0h.transpose(1, 2)  # [B, hidden, N]

        # Edge features via coboundary
        X1 = self.edge_coboundary(self.mesh.B1.t(), X0h) # [B, hidden, E]
        
        # Triangle features (if any)
        X2 = None
        X2h = None  # Initialize X2h as None early
        triangles = getattr(data, 'triangles', None)
        
        # Only process triangles if they exist and are non-empty
        if (triangles is not None and 
            triangles.ndim == 2 and 
            triangles.shape[1] == 3 and 
            triangles.shape[0] > 0):  # Check for non-empty triangles
            
            X2 = self.tri_coboundary(self.mesh.B2.t(), X1)  # [B, hidden, T]
            
            if self.enc2:
                X2 = X2.transpose(1, 2)  # [B, T, hidden]
                num_triangles = X2.size(1)
                if num_triangles > 0:  # Extra safety check
                    X2h = self.enc2(X2.reshape(-1, self.hidden))  # [B*T, hidden]
                    X2h = X2h.view(B, num_triangles, -1).transpose(1, 2)  # [B, hidden, T]

        # Encode higher-order features - reshape for Linear layers
        if self.enc1 and X1 is not None:
            X1 = X1.transpose(1, 2)                           # [B, E, hidden]
            X1h = self.enc1(X1.reshape(-1, self.hidden))      # [B*E, hidden]
            num_edges = X1.size(1)
            X1h = X1h.view(B, num_edges, -1).transpose(1, 2)  # [B, hidden, E]
        else:
            X1h = None

        # Process through simplicial layers
        bundled = [X0h]
        for _ in range(self.temporal_steps-1):
            # Process features, handling None case for X2h
            X0h_next, X1h_next, X2h_next = self.processor(X0h, X1h, X2h)
            X0h, X1h, X2h = X0h_next, X1h_next, X2h_next
            bundled.append(X0h)
        
        # Temporal projection
        temporal_concat = torch.cat(bundled, dim=1)  # [B, hidden*temporal_steps, N]
        temporal_features = temporal_concat.transpose(1, 2)  # [B, N, hidden*temporal_steps]
        temporal_features = self.temporal_proj(temporal_features)  # [B, N, hidden]
        temporal_features = temporal_features.transpose(1, 2)  # [B, hidden, N]
        
        # Decode to physical space - reshape to match target shape
        output = self.decoder(temporal_features)  # [B, time_steps, N]
        
        # Reshape to match target shape [B*N, time_steps]
        B, T, N = output.shape
        return output.transpose(1, 2).reshape(-1, T)  # [B*N, time_steps]


def build_scn_maps(mesh, max_order: int = 2):
    comp = build_complex_from_edge_index(mesh.edge_index, max_order=max_order)

    maps = {
        'A01': comp['A01'].to(torch.float32)
    }
    if max_order >= 2:
        maps.update({
            'A02': comp['A02'].to(torch.float32),
            'A12': comp['A12'].to(torch.float32),
            'triangles': comp['triangles'].to(torch.float32),
        })

    return maps

def normalize_boundary(B: torch.Tensor):
    row_sum = torch.sparse.sum(B, dim=1).to_dense()
    col_sum = torch.sparse.sum(B, dim=0).to_dense()
    
    row_norm = 1 / torch.sqrt(row_sum + 1e-8)
    col_norm = 1 / torch.sqrt(col_sum + 1e-8)

    indices = B._indices()
    values = B._values() * row_norm[indices[0]] * col_norm[indices[1]]
    return torch.sparse_coo_tensor(indices, values, B.shape)


#__________________________________________________________________________________________________________
# The following classes are used to process simplicial complexes over time.
# including a temporal simplicial processor, a stable simplicial convolution, enforcing conservation laws
class TemporalSimplicialProcessor(nn.Module):
    # This class processes simplicial complexes over time.
    # It uses a GRU to evolve the simplicial complex features over time.
    # The input features are expected to be in the form of node features (X0),
    # edge features (X1), and triangle features (X2).
    # The time information is embedded and added to the node features.
    # The simplicial complex is processed using a SimplicialProcessor,
    # and the temporal evolution is handled by a GRU.
    # The output is the updated node features (X0_temporal),
    # edge features (X1_next), and triangle features (X2_next).
    def __init__(self, hidden_dim, boundary_maps):
        super().__init__()
        self.time_embedding = nn.Linear(1, hidden_dim)
        self.processor = SimplicialProcessor(hidden_dim, boundary_maps)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim)
    
    def forward(self, X0, X1, X2, t):
        # Embed time information
        t_emb = self.time_embedding(t.unsqueeze(-1))
        X0 = X0 + t_emb.unsqueeze(-1)  # add time info to nodes
        
        # Process simplicial complex
        X0_next, X1_next, X2_next = self.processor(X0, X1, X2)
        
        # Apply temporal evolution
        X0_temporal, _ = self.temporal_gru(X0_next.transpose(1, 2))
        return X0_temporal.transpose(1, 2), X1_next, X2_next

class StableSimplicialConvolution(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = SimplicialConvolution(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, B=None):
        identity = x
        x = self.conv(x, B)
        x = self.norm(x)
        x = self.dropout(x)
        return x + identity

class PhysicsInformedProcessor(SimplicialProcessor):
    def __init__(self, hidden_dim, boundary_maps):
        super().__init__(hidden_dim, boundary_maps)
        self.conservation_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, X0, X1, X2):
        X0_next, X1_next, X2_next = super().forward(X0, X1, X2)
        
        # Enforce conservation laws through projection
        mass = self.conservation_proj(X0_next.transpose(1, 2)).sum(dim=1)
        X0_next = X0_next * (mass.unsqueeze(1) / mass.sum())
        return X0_next, X1_next, X2_next

class MultiScaleSCN(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 128]):
        super().__init__()
        self.encoders = nn.ModuleList([
            SimplicialProcessor(h_in, h_out) 
            for h_in, h_out in zip(hidden_dims[:-1], hidden_dims[1:])
        ])
        self.decoders = nn.ModuleList([
            SimplicialProcessor(h_out, h_in)
            for h_in, h_out in zip(hidden_dims[:-1], hidden_dims[1:])
        ])
