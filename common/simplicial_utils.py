# common/simplicial_utils.py
from __future__ import annotations
import networkx as nx
import torch
from torch_geometric.data import Data

# ---------- low‑level helper --------------------------------
def _normalize_incidence(rows, cols, n_rows, n_cols):
    """Return  D_r^{-1/2} · A · D_c^{-1/2}  as sparse tensor + the two degree vectors."""
    v = torch.ones(len(rows), dtype=torch.float32)
    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), v, (n_rows, n_cols)
    ).coalesce()

    deg_r = torch.sparse.sum(A, dim=1).to_dense().add_(1e-8).pow_(-0.5)
    deg_c = torch.sparse.sum(A, dim=0).to_dense().add_(1e-8).pow_(-0.5)
    vals  = deg_r[rows] * v * deg_c[cols]          # ← fixed here

    A_norm = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (n_rows, n_cols)
    ).coalesce()
    return A_norm, deg_r, deg_c

# ---------- main entry --------------------------------------
def build_complex_from_edge_index(edge_index: torch.Tensor, max_order: int = 2):
    """
    Takes a PyG Data with .edge_index and .num_nodes
    Adds .A01 .A02 .A12 .Z10 .Z20 .triangles
    Returns the same Data (for chaining).
        
    Parameters
    edge_index : LongTensor shape [2,E]  undirected
    max_order  : 1 ➡︎ only edges; 2 ➡︎ edges + triangles

    Returns:
    dict  (all torch tensors ‑ sparse where appropriate)
        'A01' |V|×|E|  edge‑to‑node
        'Z10' diag vec (|V|)
        —— if max_order >= 2 —————
        'A02' |V|×|T|  tri‑to‑node
        'A12' |E|×|T|  tri‑to‑edge
        'Z20' diag vec (|V|)
        'triangles' [n_tri,3]  LongTensor
    """
    # 0.  build graph
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())
    n_nodes = G.number_of_nodes()

    # 1.  edges
    edges = list(G.edges())                # list of (u,v)
    if len(edges) == 0:
        raise ValueError("Graph has no edges → cannot build complex")

    rows_e = torch.tensor([u for u, v in edges] + [v for u, v in edges], dtype=torch.long)
    cols_e = torch.tensor(list(range(len(edges))) * 2, dtype=torch.long)

    A01, Z10, _ = _normalize_incidence(rows_e, cols_e, n_rows=n_nodes, n_cols=len(edges))
    out = dict(A01=A01, Z10=Z10)

    # 2.  optionally triangles
    if max_order < 2:
        out['A02'] = out['A12'] = out['Z20'] = None
        out['triangles'] = torch.empty(0, 3, dtype=torch.long)
        return out
    # finds all 3-cliques (triangles) in the undirected graph, matching the SCNN definition of 2-simplices
    tris = [tuple(sorted(c)) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    tris = list(dict.fromkeys(tris))   # unique & stable order

    if len(tris) == 0:
        # supply empty placeholders so the rest of the pipeline does not
        # branch on `None`
        out.update(
            A02 = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long),
                                        torch.empty(0),
                                        (n_nodes, 0)),
            A12 = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long),
                                        torch.empty(0),
                                        (len(edges), 0)),
            Z20 = torch.zeros(0),
            triangles=torch.empty(0, 3, dtype=torch.long),
        )
        return out

    tris_t = torch.tensor(tris, dtype=torch.long)   # [n_tri,3]
    n_tri  = tris_t.size(0)
    out['triangles'] = tris_t

    # 2.a  tri → node
    rows_tn = tris_t.view(-1)
    cols_tn = torch.repeat_interleave(torch.arange(n_tri), 3)
    
    A02, Z20, _ = _normalize_incidence(rows_tn, cols_tn,
                                       n_rows=n_nodes,
                                       n_cols=n_tri)
    out.update(A02=A02, Z20=Z20)

    # 2.b  tri → edge
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    rows_te, cols_te = [], []

    for t_idx, (i, j, k) in enumerate(tris):
        for e in [(i, j), (j, k), (k, i)]:
            rows_te.append(edge_to_idx[tuple(sorted(e))])
            cols_te.append(t_idx)       

    A12, _, _ = _normalize_incidence(torch.tensor(rows_te), 
                                     torch.tensor(cols_te), 
                                     n_rows=len(edges),
                                     n_cols=n_tri)
    out['A12'] = A12
    return out

# ---------- mesh → Data enrich ------------------------------------------
def enrich_pyg_data_with_simplicial(data: Data,
                                    max_order: int = 2) -> Data:
    """
    Take a PyG `Data` object with at least `.edge_index`
    and add triangles + incidence matrices (sparse) as attributes.
    Returns *the same object* for chaining.
    """
    comp = build_complex_from_edge_index(data.edge_index, max_order)

    data.triangles = comp['triangles']     # [n_tri,3]
    data.A01       = comp['A01']           # |V|×|E|
    data.Z10       = comp['Z10']           # |E|
    if max_order >= 2:
        data.A02       = comp['A02']       # |V|×|T|
        data.A12       = comp['A12']       # |E|×|T|
        data.Z20       = comp['Z20']       # |T|
        data.triangles = comp['triangles'] # [n_tri,3]
        
    data.B1 = data.A01                     # edge -> node (1‑boundary)
    data.B2 = data.A12                     # tri  -> edge (2‑boundary)
    
    return data


#_________________________functions from snn/chebyshev.py________________________________________________________
# orig problems: mixes SciPy sparse matrices with PyTorch tensors, leading to incompatible operations

# Added sparse_scipy_to_torch: Converts SciPy sparse matrices to PyTorch sparse tensors.
# Modified normalize: Returns a PyTorch sparse tensor instead of SciPy matrix.
# Fixed Matrix Multiplication: Replaced L.mm() with torch.sparse.mm() for proper PyTorch operations.
# Adjusted Tensor Dimensions: Ensured proper tensor reshaping and concatenation.

import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

def sparse_scipy_to_torch(mat):
    """Convert a SciPy COO to a coalesced PyTorch sparse tensor."""
    mat = mat.tocoo()
    idx = np.vstack((mat.row, mat.col))
    vals = mat.data
    indices = torch.from_numpy(idx).long()
    values  = torch.from_numpy(vals).float()
    return torch.sparse_coo_tensor(indices, values, mat.shape).coalesce()

# sccn orig: Input: Expects a SciPy sparse matrix (scipy.sparse), 
# typically in CSR or COO format.
# Operates directly on the SciPy sparse matrix without any conversion.

# this: Input: Expects a PyTorch sparse tensor (torch.sparse_coo_tensor), suitable for PyTorch-based workflows.
# Converts the PyTorch sparse tensor to a SciPy COO matrix for eigenvalue computation, 
# then converts it back to a PyTorch sparse tensor.
# Set k=1 and which='LM' to find the largest magnitude eigenvalue.

# shared Normalization Logic
# Half-Interval Normalization (half_interval=True): Scales the Laplacian matrix by dividing by topeig.
# Full-Interval Normalization (half_interval=False): Scales by 2.0 / topeig and adjusts the diagonal to ensure the spectrum lies within a specific interval, often [-1, 1].
def normalize(L, half_interval=False):
    """
    Normalize a PyTorch sparse Laplacian by its largest eigenvalue.
    Converts L→SciPy, does eigsh, rescales, then converts back.
    """
    assert L.is_sparse, "normalize() expects a PyTorch sparse tensor"

    # Convert PyTorch sparse tensor to SciPy sparse COO matrix
    Lc = L.coalesce()
    idx, vals = Lc._indices().cpu().numpy(), Lc._values().cpu().numpy()
    L_scipy = sp.coo_matrix((vals, (idx[0], idx[1])), shape=L.shape)

    # Compute top eigen values
    topeig = spl.eigsh(L_scipy, k=1, which="LM", return_eigenvectors=False)[0]
    ret = L_scipy.copy()
    
    if half_interval:
        ret.data *= (1.0 / topeig)
    else:
        ret.data *= (2.0 / topeig)
        ret.setdiag(ret.diagonal() - np.ones(ret.shape[0], dtype=ret.dtype))
    assert np.all(np.isfinite(ret.data)), "L must be finite"

    # Return as PyTorch sparse
    return sparse_scipy_to_torch(ret)


def assemble(K, L, x):
    B, C_in, M = x.shape
    assert L.shape == (M, M)
    assert K > 0

    X = []
    for b in range(B):
        X123 = []
        for c_in in range(C_in):
            X23 = []
            x0 = x[b, c_in, :].unsqueeze(1)  # (M, 1)
            X23.append(x0)

            if K > 1:
                # Use PyTorch sparse matrix multiplication
                x1 = torch.sparse.mm(L, x0)
                X23.append(x1)
            for k in range(2, K):
                x_next = 2 * torch.sparse.mm(L, X23[k-1]) - X23[k-2]
                X23.append(x_next)

            X23 = torch.cat(X23, dim=1)  # (M, K)
            X123.append(X23.unsqueeze(0))
        
        X123 = torch.cat(X123, dim=0)  # (C_in, M, K)
        X.append(X123.unsqueeze(0))
    
    X = torch.cat(X, dim=0)  # (B, C_in, M, K)
    return X

#_________________________functions from snn/scnn.py________________________________________________________

# The scnn.py code is compatible with the corrected chebyshev.py as long as all input matrices (L, D) 
# are converted to PyTorch sparse tensors. Here's the verification:

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

def coo2tensor(A):
    assert is_sparse(A), "Input must be PyTorch sparse tensor"
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)

def is_sparse(x):
        return isinstance(x, torch.Tensor) and x.is_sparse
    
class SimplicialConvolution(nn.Module):
    def __init__(self, K, C_in, C_out, enable_bias = True, variance = 1.0, groups = 1):
        assert groups == 1, "Only groups = 1 is currently supported."
        super().__init__()
        assert(C_in > 0)
        assert(C_out > 0)
        assert(K > 0)
        
        self.C_in = C_in
        self.C_out = C_out
        self.K = K
        self.enable_bias = enable_bias
        self.theta = nn.parameter.Parameter(variance*torch.randn((self.C_out, self.C_in, self.K)))
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0
    
    def forward(self, L, x):
        assert(len(L.shape) == 2)
        assert(L.shape[0] == L.shape[1])
                
        (B, C_in, M) = x.shape
     
        assert(M == L.shape[0])
        assert(C_in == self.C_in)

        X = assemble(self.K, L, x)
        y = torch.einsum("bimk,oik->bom", (X, self.theta))
        assert(y.shape == (B, self.C_out, M))

        return y + self.bias

# This class does not yet implement the
# Laplacian-power-pre/post-composed with the coboundary. It can be
# simulated by just adding more layers anyway, so keeping it simple
# for now.
#
# Note: You can use this for a adjoints of coboundaries too. Just feed
# a transposed D.

'''
class takes an incidence matrix D 
(like B1, B2, or B1.T) and applies a learnable linear transform.
'''
class Coboundary(nn.Module):
    def __init__(self, C_in, C_out, enable_bias = True, variance = 1.0):
        super().__init__()

        assert(C_in > 0)
        assert(C_out > 0)

        self.C_in = C_in
        self.C_out = C_out
        self.enable_bias = enable_bias
        self.theta = nn.parameter.Parameter(variance*torch.randn((self.C_out, self.C_in)))
        
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0

    def forward(self, D, x):
        assert(len(D.shape) == 2)
        (B, C_in, M) = x.shape
        assert(D.shape[1] == M)
        assert(C_in == self.C_in)
        N = D.shape[0]

        x = x.to(dtype=D.dtype)
        X = []
        for b in range(B):
            X12 = []
            for c_in in range(self.C_in):
                X12.append(D.mm(x[b, c_in, :].unsqueeze(1)).transpose(0,1)) # (1, N)
            X12 = torch.cat(X12, 0)  # (C_in, N)
            assert X12.shape == (self.C_in, N)
            X.append(X12.unsqueeze(0))  # (1, C_in, N)
        X = torch.cat(X, dim=0)  # (B, C_in, N)
        assert X.shape == (B, self.C_in, N)

        y = torch.einsum("oi,bin->bon", (self.theta, X))
        assert y.shape == (B, self.C_out, N)
        return y + self.bias


def normalize_boundary(B: torch.Tensor):
    """Normalize boundary operator for stable training."""
    B = B.coalesce()

    row_sum = torch.sparse.sum(B, dim=1).to_dense()
    col_sum = torch.sparse.sum(B, dim=0).to_dense()
    row_norm = 1.0 / torch.sqrt(row_sum + 1e-8)
    col_norm = 1.0 / torch.sqrt(col_sum + 1e-8)

    indices = B._indices()
    values = B._values() * row_norm[indices[0]] * col_norm[indices[1]]
    
    return torch.sparse_coo_tensor(indices, values, B.shape).coalesce()


#__________________________________________________________________________________________________________
### The following function computes the Hodge Laplacian for 0-forms (nodes) and optionally for 1-forms (edges) and 2-forms (triangles).
# It uses the boundary operators B1 and B2 to compute the Laplacians.
# The function returns the Laplacians as sparse tensors.
# The Hodge Laplacian is a differential operator that combines the exterior derivative and codifferential operators.
# It is used in the context of simplicial complexes and is important for applications in computational topology and geometry processing.
def compute_hodge_laplacian(B1, B2=None):
    """Compute Hodge Laplacians for different k-forms"""
    L0 = torch.sparse.mm(B1, B1.t())  # 0-form Laplacian (nodes)
    if B2 is not None:
        L1 = (torch.sparse.mm(B1.t(), B1) + 
              torch.sparse.mm(B2, B2.t()))  # 1-form Laplacian (edges)
        L2 = torch.sparse.mm(B2.t(), B2)    # 2-form Laplacian (triangles)
        return L0, L1, L2
    
    return L0
