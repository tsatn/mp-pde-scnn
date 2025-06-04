# Message‑Passing & Simplicial Neural PDE Solvers  

## Recent Updates

- **3-Simplices Support**: Added tetrahedra processing capabilities to extend beyond triangular faces
- **Enhanced Boundary Operators**: Implemented B₃ operator for tetrahedra-triangle relationships
- **Improved Feature Processing**: Extended neural architecture to handle 3-simplex features

## Architecture Overview
- Input: A simplicial complex (mesh) with features on nodes/edges/triangles/tetrahedra.
- Encoding: Node/edge/triangle/tetrahedra features are projected into latent space via enc0, enc1, enc2, enc3.
- Processing: Simplicial convolutions propagate information across simplices using boundary operators (B1, B2, B3).
- Temporal Bundling: Aggregates features across multiple timesteps.
- Decoding: Maps processed features back to physical space.

### Boundary Operators (B₁, B₂, B₃):
```python
def build_complex_from_edge_index(edge_index, max_order=3):
    # Creates boundary operators:
    # B₁: C₁ → C₀ (edges → nodes)
    # B₂: C₂ → C₁ (triangles → edges)
    # B₃: C₃ → C₂ (tetrahedra → triangles)
```

### Feature Dimensions:
```text
Nodes (0-simplices): [B, hidden, N]
Edges (1-simplices): [B, hidden, E]
Triangles (2-simplices): [B, hidden, T]
Tetrahedra (3-simplices): [B, hidden, H]
```

### Model Processing Pipeline:
```text
Input Features
↓
Encode (enc0, enc1, enc2, enc3)
↓
Simplicial Convolutions + Boundary Operations
↓
Feature Aggregation across all simplicial orders
↓
Temporal Processing
↓
Output Prediction
```

The implementation supports fallback to lower-order simplices when higher-order structures are not available, maintaining backward compatibility with existing models.

---

## Introduction

Simplicial Physics-Informed Neural PDE Solver is a novel framework designed to solve partial differential equations (PDEs) on complex domains, particularly targeting fluid dynamics and physical systems. By leveraging the mathematical foundations of graph Hodge Laplacians, discrete exterior calculus, and simplicial complexes, this model captures geometric and topological features beyond traditional graph neural networks. The model enforces key physics laws—such as conservation of mass, momentum, and energy—directly within its architecture, making it ideal for applications that demand both high accuracy and physical plausibility. Its innovative use of multi-order interactions including nodes, edges, faces, and higher-order simplices allows for development of superior models of complex flows, vorticity, and conservation laws, surpassing conventional GNN and physics-informed neural network (PINN) approaches.

---

## High‑level Pipeline

#### Modular design: numerical solver ↔ HDF5 dataset ↔ graph/simplicial creator ↔ models ↔ training scripts.

```text
Numerical solver (WENO/FDM)  ─>  *.h5  ─>  HDF5Dataset (HDF5)  ─>  Simplical GraphCreator (Graph)
                    (generate/)        (common/utils.py)      (→ PyG Data)
                                            │
                     ┌──────────────────────────────────────┐
                     │  Neural model (experiments/models_*) │
                     │                                      │
                     │  • CNN / Res‑1D‑CNN                  │
                     │  • GNN (chain graph)                 │
                     │  • + SCN‑GNN (simplicial complex GNN)│
                     └──────────────────────────────────────┘
                                            │
                            rollout / evaluation (PDE prediction)
```


### Architecture Overview
- Input: A simplicial complex (mesh) with features on nodes/edges/triangles.
- Encoding: Node/edge/triangle features are projected into a latent space via enc0, enc1, enc2.
- Processing: Simplicial convolutions propagate information across simplices using boundary operators (B1, B2). Combines features from adjacent simplices (e.g., node features are updated using edges and triangles).
- Temporal Bundling: Aggregates features across multiple timesteps to model dynamics.
- Decoding: Maps processed features back to physical space (e.g. PDE solution).


## Core Components

### 0. Numerical PDE Data Generation 
#### generate_data.py
- Only generates raw PDE solutions in grid format
- Stores them in HDF5 files
- No simplicial structure at this stage

  ##### Input → Numerical Solution → HDF5 Storage
  1. Burgers/KdV/Wave equations solved using:
    - WENO scheme for flux terms
    - Chebyshev spectral methods for spatial derivatives 
    - Runge-Kutta (Dopri45) for time integration

  2. Data shapes at each step:
    - Base resolution: [nt=250, nx=100]
    - Super resolution: [nt=250, nx=200]
    - Stored in HDF5 with multiple resolutions
  

### 1. PDE Definitions and Numerical Methods
#### coefficients.py
- (Contains numerical coefficients for finite differences and WENO reconstructions.)
- Stores numerical constants and coefficients used for finite difference methods (FDM) and WENO (Weighted Essentially Non-Oscillatory) reconstruction methods.
- Defines specific derivative approximations (1st to 4th derivatives) using finite difference coefficients.
#### derivatives.py
- (Implements FDM and WENO derivative reconstruction methods. Numerical solvers. Uses weighted essentially non-oscillatory (WENO) and finite-difference (FDM) schemes (in common/derivatives.py) and Runge–Kutta timestepping (temporal/tableaux.py, solvers.py) to generate high-accuracy trajectories.)
- Finite Difference Method (FDM) for derivative computation (1st-4th derivatives).
- WENO5 method for spatial derivative reconstruction, including Godunov and Lax-Friedrichs flux reconstructions.
#### solvers.py:
- Implements generic PDE solvers utilizing Runge-Kutta methods defined in tableaux.py.
#### tableaux.py:
- Contains Butcher tableaux for explicit Runge-Kutta methods, e.g., Euler, Midpoint, RK4, Dopri45.
#### PDEs.py (References methods from derivatives.py for spatial reconstructions.)
- Defines PDE classes with numerical solvers (CE and WE classes):
- Combined Equation (CE): Includes special cases like Burgers and KdV equations, solved using WENO and FDM reconstructions.
- Wave Equation (WE): Implements the second-order PDE as a first-order augmented system, using Chebyshev pseudo-spectral methods.


### 2. Simplicial Complexes, Data loading and graph construction: 
#### environment.sh
- shell script setting up a Conda environment named mp-pde-solvers. Installs dependencies such as Python 3.8, PyTorch, PyTorch Geometric, CUDA toolkit, numpy, scipy, h5py, etc.
  
#### common/simplicial_utils.py (corresponds to Section 3.1 in the IEEE TPAMI paper)
- Manages the conversion of graph structures into simplicial complexes using PyTorch and PyTorch Geometric.
- Functions to normalize incidence matrices `_normalize_incidence`.
- Functions to construct simplicial complexes (nodes, edges, triangles) from PyTorch Geometric edge indices (build_complex_from_edge_index).
- Functions for Chebyshev polynomial computations adapted from Simplicial Neural Networks (SCNN), ensuring compatibility with PyTorch sparse operations.

### Simplicial Complex Creation (common/simplicial_utils.py)

#### common/simplicial_utils.py
- `build_complex_from_edge_index` and `enrich_pyg_data_with_simplicial` construct incidence (boundary) matrices B1, B2 and extract triangles, enabling SCNN layers to propagate on nodes, edges, and faces.
- a corrected Chebyshev‐based Laplacian normalization for SCNN.

* Boundary Operators (B₁, B₂):
```python
def build_complex_from_edge_index(edge_index, max_order=2):
    # Creates boundary operators B₁: C₁ → C₀ and B₂: C₂ → C₁
    # Shape B₁: [num_nodes, num_edges]
    # Shape B₂: [num_edges, num_triangles]

def compute_hodge_laplacian(B1, B2):
    L0 = torch.sparse.mm(B1, B1.t())                                 (Δ₀ = B₁B₁ᵀ) 
    L1 = torch.sparse.mm(B1.t(), B1) + torch.sparse.mm(B2, B2.t())   (Δ₁ = B₁ᵀB₁ + B₂B₂ᵀ)
```

#### common/utils.py 
- HDF5Dataset: loads low- and high-resolution trajectories, downsamples “super” resolution via conv kernels, and returns PDE parameters per sample.
- GraphCreator: slides a temporal window over trajectories, builds a 1D chain graph (via torch_cluster for radius‐ or k-NN), and packs node features (history + coordinates) and labels into a PyTorch Geometric Data object.

Graph Creation (NN)
```python
    def training_loop():
    # Creates graph structure:
    # 1. Builds radius graph (neighbors=6)
    # 2. Creates boundary matrices B1, B2
    L0 = torch.sparse.mm(graph.B1, graph.B1.transpose(0, 1)).coalesce()  # Laplacian warning
```

### 3. Machine Learning Model Architecture Implementations (CNN/GNN/SCN):
#### models_cnn.py:
- Implements a baseline ResCNN model with convolutional layers and skip connections.
- A Res-style 1D CNN stacks 8 conv layers with ELU and skip connections to predict next-window increments.
#### models_gnn.py:
- Implements Message Passing Neural Networks (MP-PDE Solver) for graph-based PDE solutions.
- MP_PDE_Solver: embeds each node’s time-window + pos/param scalars, runs multiple GNN_Layer message-passing steps over the spatial graph, then decodes via a small CNN.
#### models_gnn_snn.py:
- Implements Simplicial Convolutional Neural Networks (SCNN), leveraging simplicial complexes.
- Builds on SCNN: learns on node/edge/triangle features via custom SimplicialConvolution and Coboundary modules, aggregates via a SimplicialProcessor, bundles temporally, and decodes back to physical space.

##### Model Core: SCNPDEModel
* Receives a PyG Data object.
* Passes node features and optionally edge/triangle features through:
** Input MLP: enc0, enc1, enc2 depending on simplex level
** Simplicial convolution layers (like SimplicialConvolution)
** Output MLP to return prediction ŷ with shape [B*nx, tw]

Temporal Bundling is implemented in SCNPDEModel 
- Aggregates multiple timesteps into a single forward pass
- Uses concatenation of feature vectors across time
- Implemented via the bundled list and temporal_concat operation
- Additional temporal processing is handled by TemporalSimplicialProcessor

* Pushforward Method is implemented in GraphCreator
- Uses predictions as inputs for next timestep
- Maintains temporal consistency
- Implemented in create_next_graph
  
```text
Input: [batch, time_window, nx]
↓
Bundled Features: [batch, hidden*temporal_steps, num_nodes]
↓
Temporal Processing: [batch, hidden, num_nodes]
↓
Output: [batch, time_window, nx]
```

##### Simplicial Convolution:
```python
class SimplicialConvolution(nn.Module):
    def forward(self, x_src, B):
        # Implements σ(θ * x) where θ are learnable parameters
        # Shape x_src: [batch, channels, num_nodes/edges/triangles]

class SimplicialProcessor(nn.Module):
    def forward(self, X0, X1, X2):
        # X0: [B, hidden, N] - node features
        # X1: [B, hidden, E] - edge features 
        # X2: [B, hidden, T] - triangle features
```

##### Green's function -- implementation 
- The Hodge decomposition (in SpectralSimplicialOperator):
```python
class SpectralSimplicialOperator(nn.Module):
    def compute_operators(self, B1, B2):
        # Implements discrete version of Hodge decomposition
        # Relates to Green's function through the inverse of Laplacian
```
- The kernel integral operator from the Graph Kernel paper:
```python
# In SimplicialProcessor:
def forward(self, X0, X1, X2):
    # Implements a discrete approximation of:
    # K(x,y) = κ(x,y)v(y) where κ is learned from data
```

##### Physical Conservation Laws -- implementation (new)
```python
class PhysicsInformedProcessor(SimplicialProcessor):
    def forward(self, X0, X1, X2):
        # Enforces conservation laws through projection
        mass = self.conservation_proj(X0_next.transpose(1, 2)).sum(dim=1)
        X0_next = X0_next * (mass.unsqueeze(1) / mass.sum())
```

### 4. Training and Data Handling
- train_helper.py: Provides core training loops and evaluation methods.
- train.py: Orchestrates dataset handling, model training, evaluation, and logging.
- generate_data.py: Generates PDE datasets using numerical solutions for different PDE tasks. Combined_equation and wave_equation functions produce train/valid/test .h5 files containing solutions at multiple spatial resolutions and store PDE parameters (e.g. α, β, γ for CE; BCs and wave speed for WE).
- *temporal/solvers.py*: General PDE solver classes leveraging various numerical temporal methods.
- *temporal/tableaux.py*: Implements Butcher tableaux for explicit Runge-Kutta time integrators.

### 5. Numerical Benchmarking (using JAX)
Standalone JAX scripts such as burgers_E1_E2.py, wave_WE1.py reproduce Table 1 runtimes and errors for purely numerical solvers.
- burgers_E1_E2.py: Numerical benchmarks for the 1D Burgers' PDE equation.
- wave_WE1.py: Numerical benchmarks for the 1D Wave PDE.

## Key strengths & innovations
- Topological enrichment: Leveraging simplicial complexes injects higher-order interactions (edges→nodes, triangles→edges).
- Temporal bundling: Both GNN and SCN architectures incorporate multiple past timesteps in a single forward pass, capturing dynamics without full recurrence.
- Modular numerics --> ML pipeline: Clear separation between high-fidelity data generation and learned surrogates, making it extensible to other PDEs or dimensions.

## Git: large files
Keep mp_pde_env/ and any libtorch*.dylib in .gitignore.
With >100 MB assets, install Git‑LFS: git lfs install && git lfs track '*.dylib'.


## Usage
0. **Set up conda environment**
source environment.sh
- environment.sh: Conda environment setup script for reproducibility.
- setup.py: Python package setup script.

1. **Data Generation**
PYTHONPATH=. python generate/generate_data.py --experiment WE1 \
       --train_samples 2048 --valid_samples 128 --test_samples 128 \
       --device cpu

1. **Training**
#### nx = 100
cd ../mp-pde-scnn
PYTHONPATH=. python -m experiments.train \
  --model SCN --experiment WE1 \
  --base_resolution 250,100 --neighbors 6 --time_window 25 \
  --batch_size 16 --device cpu

#### nx = 50
python -m experiments.train \
  --model SCN --experiment WE1 \
  --base_resolution 250,50 --neighbors 6 --time_window 25 \
  --batch_size 16 --device cpu

#### nx = 40
python -m experiments.train \
  --model SCN --experiment WE1 \
  --base_resolution 250,40 --neighbors 6 --time_window 25 \
  --batch_size 16 --device cpu
<!-- 
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
 -->


## Technical Specifications

1. **Input Processing**
- Supports various PDE types (Wave, Burgers, KdV)
- Handles multiple resolutions
- Automatic simplicial complex construction

2. **Model Architecture**
- Modular design for easy extension
- Built-in stability measures
- Physics-informed constraints

3. **Output Generation**
- Multi-step prediction capability
- Conservation law preservation
- Adaptive temporal processing

4. Features
- Simplicial Complex Neural Networks: Models interactions at multiple topological orders (nodes, edges, faces).
- Hodge Laplacian Operators: Uses discrete analogues of gradient, curl, and divergence for PDE modeling.
- Physics-Informed Constraints: Enforces conservation laws and boundary conditions within the architecture or loss function.
- Flexible Domain Support: Easily adapts to 1D, 2D, and 3D domains with arbitrary geometry.
- Green’s Function Approximation: Learns fundamental solution propagation for physical systems.
- PyTorch Implementation: Modular, extensible codebase for research and real-world applications.


## Supported PDEs

- **Wave Equation**: Second-order PDE with variable wave speeds
- **Burgers Equation**: Nonlinear advection-diffusion
- **KdV Equation**: Dispersive nonlinear wave equation
- **Combined Equation**: Unified framework encompassing above cases



---


# Mathematical Foundations

## 1. Simplicial Structure

### Boundary Operators
```text
B₁: C₁ → C₀ (edges → nodes)
B₂: C₂ → C₁ (triangles → edges)
B₃: C₃ → C₂ (tetrahedra → triangles)

Dimensions:
B₁ ∈ ℝ^{N×E}  (N: nodes, E: edges)
B₂ ∈ ℝ^{E×T}  (T: triangles)
B₃ ∈ ℝ^{T×H}  (H: tetrahedra)
```

### Hodge Laplacians
```text
Δ₀ = B₁B₁ᵀ         (node Laplacian)
Δ₁ = B₁ᵀB₁ + B₂B₂ᵀ (edge Laplacian)
Δ₂ = B₂ᵀB₂ + B₃B₃ᵀ (triangle Laplacian)
Δ₃ = B₃ᵀB₃         (tetrahedra Laplacian)
```

## 2. Feature Processing

### Simplicial Convolution
For a k-simplex with features x:
```text
SCᵏ(x) = σ(θᵏx + Bₖ₊₁y)

where:
θᵏ: learnable parameters
Bₖ₊₁: boundary operator
y: features from (k+1)-simplices
σ: activation function (Swish)
```

### Feature Dimensions
```text
X₀ ∈ ℝ^{B×C×N}  (node features)
X₁ ∈ ℝ^{B×C×E}  (edge features)
X₂ ∈ ℝ^{B×C×T}  (triangle features)
X₃ ∈ ℝ^{B×C×H}  (tetrahedra features)

where:
B: batch size
C: channel dimension
```

## 3. Temporal Evolution

### Bundling Operation
```text
X̂ᵗ = concat([X⁽ᵗ⁻ᵏ⁾, ..., X⁽ᵗ⁾])
X̃ᵗ = Proj(X̂ᵗ)

where:
k: temporal window size
Proj: temporal projection layer
```

### Physics-Informed Constraints
Conservation law enforcement:
```text
M(t) = ∫ u(x,t)dx ≈ Σᵢ u(xᵢ,t)Δx
∂M/∂t = 0
```

## 4. Model Architecture

### Forward Pass
```text
Input: u(x,t) → {X₀, X₁, X₂, X₃}
↓
Encode: enc_k(Xₖ) → X̃ₖ
↓
Process: SC(X̃₀, X̃₁, X̃₂, X̃₃) → {Ŷ₀, Ŷ₁, Ŷ₂, Ŷ₃}
↓
Bundle: temporal_concat([Ŷ₀ᵗ]ₜ) → Z
↓
Output: dec(Z) → u(x,t+Δt)
```

### Green's Function Approximation
The model learns the integral kernel:
```text
u(x,t+Δt) = ∫ G(x,y)u(y,t)dy

where G is approximated by:
G(x,y) ≈ Σᵢ κ(x,y)v(y)
κ: learned kernel
v: basis functions
```

---

## Paper references
- Neural Operator: Graph Kernel Network (arXiv:2202.03376)
- Simplicial Complex Neural Networks (Wu et al., IEEE TPAMI, 2024)



<!-- 
###
Simplicial Data Generation vs Training-time Construction
* Benefits of Generating Simplicial Data During Data Generation: 
#### Computational Efficiency
Pre-computed structures reduce training time
No redundant calculations of boundary operators (B1, B2)
Consistent simplicial complexes across training runs
#### Quality Control
Can validate simplicial structure quality before training
Ensures consistent geometric features
Better control over triangle quality metrics
#### Storage Benefits
Data Storage Format:
- Nodes: [num_nodes, features]
- Edges: [num_edges, features]
- Triangles: [num_triangles, features]
- B1, B2: Pre-computed boundary operators

* Benefits of Generating Simplicial Data During Training
#### Flexibility
Can adapt mesh structure dynamically
Allows for adaptive refinement
Memory efficient (compute on-the-fly)
#### Data Augmentation
Can generate different simplicial structures for same data
Potential regularization effect
More variety in training samples
#### Best: Hybrid Solution
def generate_hybrid_simplicial_data(args):
    """Hybrid approach combining pre-computed and dynamic features"""
    
    # 1. Pre-compute core simplicial structure
    base_structure = generate_task_dataset(
        task=args.experiment,
        n_points=args.base_resolution[1]
    )
    
    # 2. Store essential components
    save_data = {
        'B1': base_structure.B1,  # Boundary operator 1
        'B2': base_structure.B2,  # Boundary operator 2
        'edge_index': base_structure.edge_index,  # Basic connectivity
    }
    
    # 3. Allow dynamic feature computation
    class DynamicFeatureBridge:
        def __init__(self, base_structure):
            self.base = base_structure
            
        def compute_features(self, data):
            # Compute dynamic features during training
            return updated_features

#### Implementation Strategy:
1. During Data Generation:
- Pre-compute and store:
- Boundary operators (B1, B2)
- Basic mesh connectivity
- Static geometric features
  
2. During Training:
- Dynamically compute:
- Time-dependent features
- Adaptive edge weights
- Dynamic simplicial features

benefits: 
- Computational efficiency from pre-computed structures
- Flexibility from dynamic feature computation
- Better memory usage
- Maintains ability to adapt during training -->