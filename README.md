# Message‑Passing & Simplicial Neural PDE Solvers  
---
## 0. Short Introduction
This GitHub repository implements a framework for solving partial differential equations (PDEs) using neural networks, specifically integrating message-passing and simplicial convolutional neural networks (SCNNs). This approach builds upon the work presented in the paper "Message Passing Neural PDE Solvers" by Brandstetter et al., which introduces neural message-passing techniques for PDE solutions .


## 1. High‑level pipeline
```text
Numerical solver (WENO/FDM)  ─▶  *.h5  ─▶  HDF5Dataset  ─▶  GraphCreator
                    (generate/)        (common/utils.py)      (→ PyG Data)
                                            │
                                            ▼
                     ┌──────────────────────────────────────┐
                     │  Neural model (experiments/models_*) │
                     │                                      │
                     │  • CNN / Res‑1D‑CNN                  │
                     │  • GNN (chain graph)                 │
                     │  • **SCN‑GNN** (simplicial branch)   │
                     └──────────────────────────────────────┘
                                            │
                                            ▼
                                   rollout / evaluation
```
### structure: 
- Input: A simplicial complex (mesh) with features on nodes/edges/triangles.
- Encoding: Node/edge/triangle features are projected into a latent space via enc0, enc1, enc2.
- Processing: Simplicial convolutions propagate information across simplices using boundary operators (B1, B2). Combines features from adjacent simplices (e.g., node features are updated using edges and triangles).
- Temporal Bundling: Aggregates features across multiple timesteps to model dynamics.
- Decoding: Maps processed features back to physical space (e.g. PDE solution).

## Input Data Flow (HDF5 → Graph)

### `HDF5Dataset`

This component loads partial differential equation (PDE) solution tensors from `.h5` files and prepares them for training.

**Functionality:**

* Loads both high-resolution and low-resolution solution data.
* Downsamples high-resolution solutions into training-compatible format `u_super`.

**Returns:**

* `u_base`: Low-resolution ground truth solution tensor of shape `[nt, nx]`.
* `u_super`: Downsampled high-resolution input tensor.
* `x`: Spatial coordinates of shape `[nx]`.
* `variables`: PDE-specific parameters (e.g., wave speed `c`, diffusion coefficient `alpha`, damping `gamma`).

### `GraphCreator`

This module constructs graph-based input data and labels from the downsampled tensors.

**Functionality:**

* Builds sliding window sequences from `u_super` for both input and target labels:
  * `data.shape` = `[B, time_window, nx]`
  * `labels.shape` = `[B, time_window, nx]`
* Constructs PyTorch Geometric `Data` objects with:
  * `x`: Node features reshaped to `[B * nx, time_window]`
  * `y`: Target outputs reshaped similarly
  * `pos`: Positional encodings `[B * nx, 2]` for time and space
  * `edge_index`: Computed via `radius_graph` or `knn_graph`
* Optionally includes:
  * PDE-specific node scalars (e.g., `bc_left`, `c`)
  * Placeholder attributes: `edge_attr`, `triangles`, and `tri_attr` for downstream use

## Model Core: SCNPDEModel
* Receives a PyG Data object.
* Passes node features and optionally edge/triangle features through:
** Input MLP: enc0, enc1, enc2 depending on simplex level
** Simplicial convolution layers (like SimplicialConvolution)
** Output MLP to return prediction ŷ with shape [B*nx, tw]


## Files
### 1. PDE Definitions and Numerical Methods
#### coefficients.py
- (Contains numerical coefficients for finite differences and WENO reconstructions.)
- Stores numerical constants and coefficients used for finite difference methods (FDM) and WENO (Weighted Essentially Non-Oscillatory) reconstruction methods.
- Defines specific derivative approximations (1st to 4th derivatives) using finite difference coefficients.
#### derivatives.py
- (Implements FDM and WENO derivative reconstruction methods.)
- Finite Difference Method (FDM) for derivative computation (1st-4th derivatives).
- WENO5 method for spatial derivative reconstruction, including Godunov and Lax-Friedrichs flux reconstructions.
#### PDEs.py (References methods from derivatives.py for spatial reconstructions.)
- Defines PDE classes with numerical solvers (CE and WE classes):
- Combined Equation (CE): Includes special cases like Burgers and KdV equations, solved using WENO and FDM reconstructions.
- Wave Equation (WE): Implements the second-order PDE as a first-order augmented system, using Chebyshev pseudo-spectral methods.
#### solvers.py:
- Implements generic PDE solvers utilizing Runge-Kutta methods defined in tableaux.py.
#### tableaux.py:
- Contains Butcher tableaux for explicit Runge-Kutta methods, e.g., Euler, Midpoint, RK4, Dopri45.


### 2. Simplicial Complexes and Graph Utilities
#### environment.sh
- shell script setting up a Conda environment named mp-pde-solvers. Installs dependencies such as Python 3.8, PyTorch, PyTorch Geometric, CUDA toolkit, numpy, scipy, h5py, etc.
#### simplicial_utils.py
- Manages the conversion of graph structures into simplicial complexes using PyTorch and PyTorch Geometric.
- Functions to normalize incidence matrices (_normalize_incidence).
- Functions to construct simplicial complexes (nodes, edges, triangles) from PyTorch Geometric edge indices (build_complex_from_edge_index).
- Functions for Chebyshev polynomial computations adapted from Simplicial Neural Networks (SCNN), ensuring compatibility with PyTorch sparse operations.


### 3. Machine Learning Models (CNN/GNN/SCN):
#### models_cnn.py:
- Implements a baseline ResCNN model with convolutional layers and skip connections.
#### models_gnn.py:
- Implements Message Passing Neural Networks (MP-PDE Solver) for graph-based PDE solutions.
#### models_gnn_snn.py:
- Implements Simplicial Convolutional Neural Networks (SCNN), leveraging simplicial complexes.


### 4. Training and Data Handling
train_helper.py: Provides core training loops and evaluation methods.

train.py: Orchestrates dataset handling, model training, evaluation, and logging.

generate_data.py: Generates PDE datasets using numerical solutions for different PDE tasks.

solvers.py: General PDE solver classes leveraging various numerical temporal methods.

tableaux.py: Implements Butcher tableaux for explicit Runge-Kutta time integrators.

### 5. Numerical Benchmarking (using JAX)
burgers_E1_E2.py: Numerical benchmarks for the 1D Burgers' PDE equation.

wave_WE1.py: Numerical benchmarks for the 1D Wave PDE.

## Git: large files
Keep mp_pde_env/ and any libtorch*.dylib in .gitignore.
With >100 MB assets, install Git‑LFS: git lfs install && git lfs track '*.dylib'.

## Running Commands
### Set up conda environment
source environment.sh
- environment.sh: Conda environment setup script for reproducibility.
- setup.py: Python package setup script.

### NEW MODEL: RUN: Produce datasets for tasks E1, E2, E3, WE1, WE2, WE3
python generate/generate_data.py --experiment WE1 \
       --train_samples 2048 --valid_samples 128 --test_samples 128 \
       --device cpu
       
python experiments/train.py \
       --model SCN --experiment WE1 \
       --base_resolution 250,100 --neighbors 6 --time_window 25 \
       --batch_size 16 --device cuda:0  --log True
       
 block                                 |  parameters 
| ------------------------------------ | ---------- |
| Encoder0 (Linear 25→128)             | 3 328      |
| Simplicial conv ×3 (0‑,1‑,2‑simplex) | \~50 k     |
| Decoder (128→25)                     | 3 225      |
| **Total**                            | **≈ 60 k** |

### OLD MODEL:
### Produce datasets for tasks E1, E2, E3, WE1, WE2, WE3
`python generate/generate_data.py --experiment={E1, E2, E3, WE1, WE2, WE3} --train_samples=2048 --valid_samples=128 --test_samples=128 --log=True --device=cuda:0`

####  Train MP-PDE solvers for tasks E1, E2, E3
`python experiments/train.py --device=cuda:0 --experiment={E1, E2, E3} --model={GNN, ResCNN, Res1DCNN} --base_resolution=250,{100,50,40} --time_window=25 --log=True`

#### Train MP-PDE solvers for tasks WE1, WE2
`python experiments/train.py --device=cuda:0 --experiment={WE1, WE2} --base_resolution=250,{100,50,40} --neighbors=6 --time_window=25 --log=True`

#### Train MP-PDE solvers for task WE3
`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,100 --neighbors=20 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,50 --neighbors=12 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,40 --neighbors=10 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,40 --neighbors=6 --time_window=25 --log=True`


# Model Comparison
Aspect	                        Your SCN-PDE Model	MP-Neural-PDE-Solvers	Simplicial-NN (SNN)
Input Features	                       - Nodes: PDE       states + coordinates.
- Edges: Edge lengths.
- Triangles: Areas.	- Nodes: PDE states + coordinates.
- Edges: Not used (implicit via adjacency).	- Nodes/Edges/Triangles: Features per simplex (e.g., node signals, edge flows, triangle pressures).
Input Operators: Boundary matrices B1 (edges→nodes), B2 (triangles→edges).	Standard adjacency matrix (edges as connectivity).	Hodge Laplacians (L0, L1, L2) for gradient/curl/divergence.
Output: Predicted PDE state (node features).	Predicted PDE state (node features).	Task-dependent (e.g., edge flows, node classifications, triangle properties).
Temporal Input: Temporal bundling of temporal_steps hidden states.	Time-unrolled PDE states passed through recurrent message passing.	No explicit temporal component (designed for static complexes).


# Improvements 
- Adopt Hodge Laplacians (from Simplicial-NN) to enrich topological interactions.
- Integrate time-unrolling (from MP-PDE) for better temporal modeling.
- Use Simplicial-NN’s normalization for improved stability.
- Hessian Matrix
- 

