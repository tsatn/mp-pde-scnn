# Message‑Passing & Simplicial Neural PDE Solvers  
---

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
## Git: large files
Keep mp_pde_env/ and any libtorch*.dylib in .gitignore.

If you must version >100 MB assets, install Git‑LFS: git lfs install && git lfs track '*.dylib'.

## Files
common/utils.py: HDF5Dataset & GraphCreator
common/simplicial_utils.py: Adds incidence matrices A01, A12, triangles, B1,B2
equations/PDEs.py: PDE base class + Burgers (CE) & Wave (WE) implementations
generate/generate_data.py: Offline local dataset creation
experiments/train.py:	CLI entry, logging, epoch loop
experiments/train_helper.py: training_loop(), test_* helpers
experiments/models_cnn.py: Res‑CNN baselines
experiments/models_gnn.py: vanilla MP‑PDE GNN
experiments/models_gnn_snn.py: SCNPDEModel with Simplicial‑Conv processor
third_party/snn/*: lightweight stubs to satisfy historical imports

## Citation
@article{brandstetter2022message,
  title   = {Message Passing Neural PDE Solvers},
  author  = {Brandstetter, Johannes and Worrall, Daniel and Welling, Max},
  journal = {ICLR},
  year    = {2022}
}

## Set up conda environment
source environment.sh

### NEW MODEL: RUN: Produce datasets for tasks E1, E2, E3, WE1, WE2, WE3
python generate/generate_data.py --experiment WE1 \
       --train_samples 2048 --valid_samples 128 --test_samples 128 \
       --device cpu
       
python experiments/train.py \
       --model SCN --experiment WE1 \
       --base_resolution 250,100 --neighbors 6 --time_window 25 \
       --batch_size 16 --device cuda:0  --log True
       
| block                                | parameters |
| ------------------------------------ | ---------- |
| Encoder0 (Linear 25→128)             | 3 328      |
| Simplicial conv ×3 (0‑,1‑,2‑simplex) | \~50 k     |
| Decoder (128→25)                     | 3 225      |
| **Total**                            | **≈ 60 k** |


### Produce datasets for tasks E1, E2, E3, WE1, WE2, WE3
`python generate/generate_data.py --experiment={E1, E2, E3, WE1, WE2, WE3} --train_samples=2048 --valid_samples=128 --test_samples=128 --log=True --device=cuda:0`

###  Train MP-PDE solvers for tasks E1, E2, E3
`python experiments/train.py --device=cuda:0 --experiment={E1, E2, E3} --model={GNN, ResCNN, Res1DCNN} --base_resolution=250,{100,50,40} --time_window=25 --log=True`

### Train MP-PDE solvers for tasks WE1, WE2
`python experiments/train.py --device=cuda:0 --experiment={WE1, WE2} --base_resolution=250,{100,50,40} --neighbors=6 --time_window=25 --log=True`

### Train MP-PDE solvers for task WE3
`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,100 --neighbors=20 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,50 --neighbors=12 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,40 --neighbors=10 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,40 --neighbors=6 --time_window=25 --log=True`

