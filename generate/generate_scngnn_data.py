import math
import h5py
import torch
from common.simplicial_utils import enrich_mesh_with_simplicial_data
from torch_geometric.data import Data

class SimplicialFeatureBridge:
    def __init__(self, complex_data):
        self.complex = complex_data
        
    def get_edge_features(self, node_coords):
        """Create edge features from node coordinates"""
        # edge_feats = []
        # for (u, v) in self.complex['1-simplices']:
        #     vec = node_coords[v] - node_coords[u]
        #     edge_feats.append(torch.tensor([vec.norm()]))
        # return torch.stack(edge_feats)
        edge_feats = torch.tensor([
            [torch.norm(node_coords[v] - node_coords[u])] 
            for u, v in self.complex['1-simplices']
        ])
        return edge_feats
    
    def get_triangle_features(self, node_coords):
        """Create triangle features using geometric properties"""
        # tri_feats = []
        # for tri in self.complex['2-simplices']:
        #     points = [node_coords[i] for i in tri]
        #     vec1 = points[1] - points[0]
        #     vec2 = points[2] - points[0]
        #     area = 0.5 * torch.cross(vec1, vec2).norm()
        #     tri_feats.append(torch.tensor([area]))
        # return torch.stack(tri_feats) if tri_feats else torch.zeros(0, 1)
        tri_feats = torch.tensor([
            [0.5 * torch.norm(torch.cross(
                node_coords[tri[1]] - node_coords[tri[0]], 
                node_coords[tri[2]] - node_coords[tri[0]]
            ))] 
            for tri in self.complex['2-simplices']
        ]) if len(self.complex['2-simplices']) else torch.zeros((0, 1))
        return tri_feats
    
def generate_task_dataset(task: str, n_points: int = 512, T: int = 11):
    """Generate enriched simplicial dataset for MP-Neural-PDE"""
    # Create base mesh
    x = torch.linspace(0, 1, n_points)
    mesh = Data(
        x=x.unsqueeze(-1),
        edge_index=torch.stack([torch.arange(n_points-1), torch.arange(1, n_points)]).long(),
        num_nodes=n_points
    )
    # Add simplicial structure
    mesh = enrich_mesh_with_simplicial_data(mesh)
    # Create features
    bridge = SimplicialFeatureBridge(mesh)
    return Data(
        x=torch.sin(2 * math.pi * x).unsqueeze(-1),  # Node features
        edge_attr=bridge.get_edge_features(mesh.x.squeeze()),
        tri_attr=bridge.get_triangle_features(mesh.x.squeeze()),
        **{k: getattr(mesh, k) for k in ['B1', 'B2', 'B1_norm', 'B2_norm']}
    )

def generate_data_wave_equation(experiment: str, boundary_condition: str, pde: dict, 
                               mode: str, num_samples: int = 1, batch_size: int = 1,
                               wave_speed: float = 2., device: torch.device = "cpu"):
    """Enhanced version with simplicial data support"""
    # Initialize HDF5 dataset
    save_name = f"data/{experiment}_{mode}.h5"
    with h5py.File(save_name, 'w') as h5f:
        dataset = h5f.create_group(mode)
        
        # Store metadata and simplicial structure
        for key in pde:
            mesh = generate_task_dataset(experiment, pde[key].grid_size[1])
            dataset.create_dataset(f'{key}/nodes', data=mesh.x.numpy())
            dataset.create_dataset(f'{key}/edges', data=mesh.edge_attr.numpy())
            dataset.create_dataset(f'{key}/triangles', data=mesh.tri_attr.numpy())
            dataset.create_dataset(f'{key}/B1', data=mesh.B1_norm.indices().numpy())
            dataset.create_dataset(f'{key}/B2', data=mesh.B2_norm.indices().numpy())
            
        # Store simulation parameters
        dataset.attrs['wave_speed'] = wave_speed
        dataset.attrs['boundary_condition'] = boundary_condition