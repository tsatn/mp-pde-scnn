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
def build_complex_from_edge_index(edge_index: torch.Tensor,
                                  max_order: int = 2):

    """
    Takes a PyG Data with .edge_index and .num_nodes
    Adds .A01 .A02 .A12 .Z10 .Z20 .triangles
    Returns the same Data (for chaining).
    """
    
    """
    Parameters
    ----------
    edge_index : LongTensor shape [2,E]  undirected
    max_order  : 1 ➡︎ only edges; 2 ➡︎ edges + triangles

    Returns
    -------
    dict  (all torch tensors ‑ sparse where appropriate)
        'A01' |V|×|E|  edge‑to‑node
        'Z10' diag vec (|V|)
        —— if max_order >= 2 ————————————————
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

    rows_e = torch.tensor([u for u, v in edges] +
                          [v for u, v in edges], dtype=torch.long)
    cols_e = torch.tensor(list(range(len(edges))) * 2, dtype=torch.long)

    A01, Z10, _ = _normalize_incidence(rows_e, cols_e,
                                       n_rows=n_nodes,
                                       n_cols=len(edges))
    out = dict(A01=A01, Z10=Z10)

    # 2.  optionally triangles
    if max_order < 2:
        out['A02'] = out['A12'] = out['Z20'] = None
        out['triangles'] = torch.empty(0, 3, dtype=torch.long)
        return out

    tris = [tuple(sorted(c))
            for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    tris = list(dict.fromkeys(tris))   # unique & stable order

    if len(tris) == 0:
        # supply empty placeholders so the rest of the pipeline does not
        # branch on `None`
        out.update(
            A02=torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long),
                                        torch.empty(0),
                                        (n_nodes, 0)),
            A12=torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long),
                                        torch.empty(0),
                                        (len(edges), 0)),
            Z20=torch.zeros(0),
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
            rows_te.append(edge_to_idx[e])
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

    data.triangles = comp['triangles']          # [n_tri,3]
    data.A01       = comp['A01']           # |V| × |E|
    data.Z10       = comp['Z10']          # |E|
    if max_order >= 2:
        data.A02       = comp['A02']         # |V| × |T|
        data.A12       = comp['A12']       # |E| × |T|
        data.Z20       = comp['Z20']      # |T|
        data.triangles = comp['triangles']  # [n_tri,3]
        
    data.B1 = data.A01            # edge -> node  (1‑boundary)
    data.B2 = data.A12            # tri  -> edge  (2‑boundary)
    
    return data

