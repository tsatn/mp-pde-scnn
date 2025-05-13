import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
from equations.PDEs import *
from common.simplicial_utils import enrich_pyg_data_with_simplicial

###
    # how u and y are built: given that the expected time_window = 25.
    # permute the tensors so that the time‑window dimension is kept
    # (25), then reshape once outside the loop — no more transposes inside the
    # loop.
###
class HDF5Dataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: PDE,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (250, 100) if base_resolution is None else base_resolution
        self.super_resolution = (250, 200) if super_resolution is None else super_resolution
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'

        ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
        ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        assert (ratio_nt.is_integer())
        assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)
        self.nt = self.data[self.dataset_base].attrs['nt']
        self.dt = self.data[self.dataset_base].attrs['dt']
        self.dx = self.data[self.dataset_base].attrs['dx']
        self.x = self.data[self.dataset_base].attrs['x']
        self.tmin = self.data[self.dataset_base].attrs['tmin']
        self.tmax = self.data[self.dataset_base].attrs['tmax']
        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
            weights = torch.tensor([[[[0.2]*5]]])
            u_super = F.conv2d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            x = self.x
            
            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['alpha'] = self.data['alpha'][idx]
            variables['beta'] = self.data['beta'][idx]
            variables['gamma'] = self.data['gamma'][idx]
            return u_base, u_super, x, variables

        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv2d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            
            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv2d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]

            return u_base, u_super, x, variables

        else:
            raise Exception("Wrong experiment")

class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 5,
                 t_resolution: int = 250,
                 x_resolution: int =100
                 ) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
            time_ration (int): temporal ratio between base and super resolution
            space_ration (int): spatial ratio between base and super resolution
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_res = t_resolution
        self.x_res = x_resolution
        # let PDE know the actual grid used
        self.pde.grid_size = (self.t_res, self.x_res)

        
        # ---- static 1‑D chain graph for all samples -----------------
        # nodes are 0 … (nx‑1); undirected edges (i,i+1)
        self.num_nodes = self.x_res
        src  = torch.arange(self.x_res - 1, dtype=torch.long)
        dst  = src + 1
        # edge_index shape [2, 2*(nx‑1)]
        self.edge_index = torch.stack(
            [torch.cat([src, dst]),          # forward edges
             torch.cat([dst, src])], dim=0   # backward edges (undirected)
        )

        # positions along the spatial axis (needed by SCN mesh)
        self._grid = torch.linspace(
            getattr(pde, 'xmin', 0.0),       # 0 if attribute missing
            getattr(pde, 'xmax', 1.0),
            self.x_res
        )
        
        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)
    
    def get_grid(self) -> torch.Tensor:
        """[num_nodes] tensor with node x‑positions (0‑1)."""
        # Return the 1‑D spatial grid (x‑coordinates) as a 1‑D tensor of
        # length nx = self.pde.grid_size[1].
        # return self._grid
        # return torch.linspace(
        #     0.0, self.pde.L, self.pde.grid_size[1],
        #     device=self.pde.device if hasattr(self.pde, "device") else "cpu"
        # )
        """Returns the 1‑D spatial coordinates as a tensor [N]."""
        return torch.linspace(0, self.pde.L, self.pde.grid_size[1])


    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels
    
    
    def create_graph(
            self,
            data: torch.Tensor,   # shape [B, time_window, nx]
            labels: torch.Tensor, # shape [B, time_window, nx]
            x: torch.Tensor,
            variables: dict,
            steps: list) -> Data:

        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        
        # B = batch size, N = number of spatial nodes, tw = time-window length
        B, tw, nx = data.size()    # Get batch size from data tensor
        nt        = self.t_res         
        t_vec = torch.linspace(self.pde.tmin, self.pde.tmax, nt, device=data.device)
        
        # positional encodings for every node in the graph
        x_pos = x[0].repeat(B)                          # [B*nx]
        t_pos = torch.cat([t_vec[s].repeat(nx) for s in steps])  # [B*nx]  ← time coord
        pos   = torch.stack([t_pos, x_pos], dim=1)      # [B*nx, 2]
        
        # ------------- features & labels ---------------------------------
        #   data:  [B, tw, nx]  →  [B, nx, tw]  →  [B*nx, tw]
        u = data.permute(0, 2, 1).reshape(-1, tw)      # [B*nx, tw]
        y = labels.permute(0, 2, 1).reshape(-1, tw)    # [B*nx, tw]
        
        # ------------- edge index (radius or k‑NN) -----------------------        
        # torch.arange(B, device=data.device) creates the batch IDs directly on the right device (CPU/GPU).
        # repeat_interleave(nx) expands each batch ID nx times, giving a vector of length B × nx, 
        # which now matches x_pos, t_pos, and the node features you stack.                
        # batch index  [B*nx]  → 0,0,…,1,1,…,B-1
        
        batch_vec = torch.arange(B, device=data.device).repeat_interleave(nx)

        edge_index = torch.cat([
             self.edge_index + i*nx for i in range(B)
        ], dim=1)                                     # [2,  B*(nx-1)*2]
        
        if f'{self.pde}' == 'CE':
            dx = x[0, 1] - x[0, 0]
            from math import exp
            r  = self.n * dx + 1 * exp(-4)
            edge_index = radius_graph(x_pos, r=r, batch=batch_vec, loop=False)

        else: # WE
            edge_index = knn_graph(x_pos, k=self.n, batch=batch_vec, loop=False)

        # —- keep each undirected edge once, oriented i < j 
        src, dst = edge_index
        mask     = src < dst             # canonical orientation
        edge_index = torch.stack([src[mask], dst[mask]], dim=0)

        # build PyG Data ---------------------------------------------------
        graph = Data(x=u, edge_index=edge_index, y=y, pos=pos, batch=batch_vec)
        graph = enrich_pyg_data_with_simplicial(graph, max_order=2)

        # ------ (optional) PDE‑specific scalars per node ------------------
        if f'{self.pde}' == 'WE':
            if 'bc_left' in variables:           # normal training path
                bc_left  = torch.tensor(variables['bc_left' ][batch_vec])
                bc_right = torch.tensor(variables['bc_right'][batch_vec])
                c_speed  = torch.tensor(variables['c'       ][batch_vec])
                graph.bc_left, graph.bc_right, graph.c = bc_left, bc_right, c_speed
                # else:  quick unit-test with {} — just skip per-node scalars
    
        # --------------------------------------------------------------
        #  SCN needs a feature tensor for every edge / triangle
        #
        E = edge_index.size(1)
        graph.edge_attr = torch.zeros(E, 1)          # 1 feature per edge
        #
        graph.triangles = torch.empty(0, 3, dtype=torch.long)  # no tris yet
        graph.tri_attr  = torch.zeros(0, 1)
        # --------------------------------------------------------------
        
        return graph



    def create_next_graph(self,
                             graph: Data,
                             pred: torch.Tensor,
                             labels: torch.Tensor,
                             steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # Output is the new input
        graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
        graph.y = y
        graph.pos[:, 0] = t_pos

        return graph
    
