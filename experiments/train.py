import argparse
import os
import copy
import sys
import time
from datetime import datetime
import torch
import argparse
# import random
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torch_geometric.nn import MessagePassing
from experiments.models_gnn import MP_PDE_Solver
from experiments.train_helper import *
from equations.PDEs import *
from torch_geometric.data import Data
from experiments.models_gnn_snn  import normalize_boundary
from common.simplicial_utils import Coboundary, normalize
from experiments.models_gnn_snn import SCNPDEModel
from common.simplicial_utils import enrich_pyg_data_with_simplicial, Coboundary, normalize
from common.utils import HDF5Dataset, GraphCreator
from experiments.models_cnn import BaseCNN
from experiments.train_helper import *
import wandb

def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'experiments/log'):
        os.mkdir(f'experiments/log')
    if not os.path.exists(f'models'):
        os.mkdir(f'models')


def train(args: argparse,
          pde: PDE,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          loader: DataLoader,
          graph_creator: GraphCreator,
          criterion: torch.nn.modules.loss,
          run,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    model.train()
    
    # Sample number of unrolling steps during training (pushforward trick)
    # Default is to unroll zero steps in the first epoch and then increase the max amount of unrolling steps per additional epoch.
    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.
    ave_loss, total_loss, loss_size = 0, 0, 0
    ave_accuracy, total_accuracy, accuracy_size = 0, 0, 0

    for i in range(graph_creator.t_res): 
        losses, accuracies = training_loop(model, unrolling, args.batch_size, optimizer, loader, graph_creator, criterion, device)
        if(i % args.print_interval == 0):
            print(f'Training Loss (progress: {i / graph_creator.t_res:.2f}): {torch.mean(losses)}')
            print(f'Training Accuracy (progress: {i / graph_creator.t_res:.2f}): {torch.mean(accuracies)}')
            loss_size += 1
            accuracy_size += 1
        total_loss += torch.mean(losses)  
        total_accuracy += torch.mean(accuracies)
    ave_loss = total_loss/loss_size
    ave_accuracy = total_accuracy/accuracy_size
    run.log({"train_loss": float(torch.mean(ave_loss)), "epoch": epoch})
    run.log({"train_accuracy": float(torch.mean(ave_accuracy)), "epoch": epoch})
    
    
def test(args: argparse,
         pde: PDE,
         model: torch.nn.Module,
         loader: DataLoader,
         graph_creator: GraphCreator,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    Test routine
    Both step wise and unrolled forward losses are computed
    and compared against low resolution solvers
    step wise = loss for one neural network forward pass at certain timepoints
    unrolled forward loss = unrolling of the whole trajectory
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model.eval()

   # first we check the losses for different timesteps (one forward prediction array!)
    steps = [t for t in range(graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
    losses = test_timestep_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    # next we test the unrolled losses
    losses = test_unrolled_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  nr_gt_steps=args.nr_gt_steps,
                                  nx_base_resolution=args.base_resolution[1],
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)
    
    return torch.mean(losses)


def main(args: argparse):
    try:
        device = args.device
        check_directory()

        base_resolution = args.base_resolution
        super_resolution = args.super_resolution

        # Check for experiments and if resolution is available
        
        # “irregular” here simply means “radius / k‑NN graphs instead of the
        # implicit Chebyshev lattice used for spectral derivatives”.
        # We are now deliberately creating such irregular graphs (with knn_graph
        # or radius_graph) so that the simplicial complex makes sense.
        # Therefore that check is obsolete for the new SCN model.
        
        if args.experiment == 'E1' or args.experiment == 'E2' or args.experiment == 'E3':
            pde = CE(device=device)
            # assert(base_resolution[0] == 250)
            assert(base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40)
        
        elif args.experiment == 'WE1' or args.experiment == 'WE2' or args.experiment == 'WE3':
            pde = WE(device=device)
            # assert (base_resolution[0] == 250)
            assert (base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40 or base_resolution[1] == 20)
            graph_creator = GraphCreator(pde, neighbors=args.neighbors,
                                 time_window=args.time_window,
                                 t_resolution=args.nt_base,
                                 x_resolution=args.nx_base)
            
        else:
            raise Exception("Wrong experiment")

        # ------------------------------------------------------------------------
        #  LOAD/CREATE DATASETS
        # ------------------------------------------------------------------------
        train_string = f'data/{pde}_train_{args.experiment}.h5'
        valid_string = f'data/{pde}_valid_{args.experiment}.h5'
        test_string  = f'data/{pde}_test_{args.experiment}.h5'
        
        try:

            train_dataset = HDF5Dataset(train_string, pde=pde, mode='train',
                                        base_resolution=[args.nt_base, args.nx_base],
                                        super_resolution=args.super_resolution)
            train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=0)

            valid_dataset = HDF5Dataset(valid_string, pde=pde, mode='valid', 
                                        base_resolution=[args.nt_base, args.nx_base], 
                                        super_resolution=args.super_resolution)
            valid_loader = DataLoader(valid_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0)

            test_dataset = HDF5Dataset(test_string, pde=pde, mode='test', 
                                       base_resolution=[args.nt_base, args.nx_base],
                                       super_resolution=args.super_resolution)
            test_loader = DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0)
            
        except FileNotFoundError as e:
            # --------------------------------------------------------------------
            # File(s) missing  →  generate them on‑the‑fly with the original script
            # --------------------------------------------------------------------
            print("Datasets not found – generating now")
            if args.experiment.startswith('WE'):        # wave equation family
                from generate.generate_data import wave_equation

                bc_map = {'WE1': 'dirichlet', 'WE2': 'neumann', 'WE3': 'mixed'}
                wave_equation(
                    experiment            = args.experiment,
                    boundary_condition    = bc_map.get(args.experiment, 'dirichlet'),
                    num_samples_train     = args.batch_size * 32,
                    num_samples_valid     = args.batch_size * 32,
                    num_samples_test      = args.batch_size * 32,
                    wave_speed            = args.wave_speed,
                    batch_size            = 1,            # generator restriction
                    device                = args.device
                )
            else:                                       # combined CE equation family
                from generate.generate_data import combined_equation
                combined_equation(
                    experiment            = args.experiment,
                    num_samples_train     = args.batch_size * 32,
                    num_samples_valid     = args.batch_size * 32,
                    num_samples_test      = args.batch_size * 32,
                    batch_size            = 4,
                    device                = args.device
                )

        # Equation specific parameters
        pde.tmin = train_dataset.tmin
        pde.tmax = train_dataset.tmax
        pde.grid_size = base_resolution
        pde.dt = train_dataset.dt

        dateTimeObj = datetime.now()
        timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'

        if(args.log):
            logfile = f'experiments/log/{args.model}_{pde}_{args.experiment}_xresolution{base_resolution[1]}-{super_resolution[1]}_n{args.neighbors}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}.csv'
            print(f'Writing to log file {logfile}')
            sys.stdout = open(logfile, 'w')

        save_path = f'models/GNN_{pde}_{args.experiment}_xresolution{base_resolution[1]}-{super_resolution[1]}_n{args.neighbors}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}.pt'
        print(f'Training on dataset {train_string}')
        print(device)
        print(save_path)

        # Equation specific input variables
        eq_variables = {}
        if not args.parameter_ablation:
            if args.experiment == 'E2':
                print(f'Beta parameter added to the GNN solver')
                eq_variables['beta'] = 0.2
            elif args.experiment == 'E3':
                print(f'Alpha, beta, and gamma parameter added to the GNN solver')
                eq_variables['alpha'] = 3.
                eq_variables['beta'] = 0.4
                eq_variables['gamma'] = 1.
            elif (args.experiment == 'WE3'):
                print('Boundary parameters added to the GNN solver')
                eq_variables['bc_left'] = 1
        #         eq_variables['bc_right'] = 1

        graph_creator = GraphCreator(
                                    pde             = pde,
                                    neighbors       = args.neighbors,
                                    time_window     = args.time_window,
                                    t_resolution    = args.nt_base,
                                    x_resolution    = args.nx_base).to(device)

        # In the model selection block of main():
        if args.model == 'GNN':
            model = MP_PDE_Solver(pde=pde,
                                time_window=graph_creator.tw,
                                eq_variables=eq_variables).to(device)
        
        elif args.model == 'SCN':
            # build a static mesh
            mesh = Data(edge_index = graph_creator.edge_index,
                        num_nodes  = graph_creator.num_nodes,
                        pos        = graph_creator.get_grid().unsqueeze(-1))   # [N,1]

            # enrich with A01 / A02 / A12 / triangles
            # attaches A01, A12, A02, B1, B2, etc. to the mesh, returns the enriched mesh object ready for use in SCNPDEModel
            enrich_pyg_data_with_simplicial(mesh, max_order=2)
            # ensure our sparse B1/B2 are coalesced for downstream SCN routines
            # PyTorch sparse incidence matrices from enrich_pyg_data_with_simplicial
            mesh.B1 = normalize_boundary(mesh.B1).coalesce()
            mesh.B2 = normalize_boundary(mesh.B2).coalesce()
            mesh.edge_attr = torch.zeros(mesh.num_edges, 1)
            mesh.tri_attr  = torch.zeros(mesh.num_triangles, 1) if hasattr(mesh, 'num_triangles') else None

            # node_in_dim = graph_creator.x_res          # 100 for nx=100
            feat_dims = {
                'node'     : graph_creator.tw + 2,   # 25 history + (t,x)
                'edge'     : 1,
                'triangle' : 1
            }
            model = SCNPDEModel(mesh, graph_creator.tw, feat_dims, hidden=128).to(device)
            
        elif args.model == 'BaseCNN':
            model = BaseCNN(pde=pde,
                            time_window=args.time_window).to(device)
            
        else:
            raise Exception("Wrong model specified")
        
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Number of parameters: {params}')

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.unrolling, 5, 10, 15], gamma=args.lr_decay)

        # Training loop
        min_val_loss = 10e30
        test_loss = 10e30
        criterion = torch.nn.MSELoss(reduction="sum").to(torch.float32)
        
        # Initialize wandb before training
        run = wandb.init(
            entity="shtian-uc-san-diego",
            project="mp-pde-scnn",  
            # name=args.experiment,
            name=f"{args.model}_{args.experiment}_nx{args.base_resolution[1]}",
            config={
                "model": args.model,
                "pde_type": args.experiment,
                "base_resolution": args.base_resolution,
                "super_resolution": args.super_resolution,
                "neighbors": args.neighbors,
                "time_window": args.time_window,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "unrolling": args.unrolling,
                "architecture": {
                    "hidden_dim": 128,
                    "temporal_steps": 3
                }
            },
            tags=["pde-solver", args.experiment, args.model]
        )
        
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch}")
            train(args, pde, epoch, model, optimizer, train_loader, graph_creator, criterion, run = run, device=device)
            print("Evaluation on validation dataset:")
            val_loss = test(args, pde, model, valid_loader, graph_creator, criterion, device=device)
            if val_loss < min_val_loss:
                print("Evaluation on test dataset:")
                test_loss = test(args, pde, model, test_loader, graph_creator, criterion, device=device)
                # Save model
                torch.save(model.state_dict(), save_path)
                print(f"Saved model at {save_path}\n")
                min_val_loss = val_loss

            scheduler.step()

        print(f"Test loss: {test_loss}")
    finally:
        wandb.finish()  # Ensure wandb run is properly closed


def parse_args():
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--experiment', type=str, default='',
                        help='Experiment for PDE solver should be trained: [E1, E2, E3, WE1, WE2, WE3]')

    # Model
    parser.add_argument('--model', type=str, default='GNN',
                        help='Model used as PDE solver: [GNN, BaseCNN]')

    # Model parameters
    parser.add_argument('--batch_size', type=int, default=16,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=1,
            help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Learning rate')
    parser.add_argument('--lr_decay', type=float,
                        default=0.4, help='multistep lr decay')
    parser.add_argument('--parameter_ablation', type=eval, default=False,
                        help='Flag for ablating MP-PDE solver without equation specific parameters')

    # Base resolution and super resolution
    # parser.add_argument('--base_resolution', type=lambda s: [int(item) for item in s.split(',')],
    #         default=[250, 100], help="PDE base resolution on which network is applied")
    # new
    parser.add_argument("--base_resolution", type=str,
            default="250,100",  help="nt,nx  (e.g. 250,100 or 250,50)")
    parser.add_argument('--super_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 200], help="PDE super resolution for calculating training and validation loss")
    parser.add_argument('--neighbors', type=int,
                        default=3, help="Neighbors to be considered in GNN solver")
    parser.add_argument('--time_window', type=int,
                        default=25, help="Time steps to be considered in GNN solver")
    parser.add_argument('--unrolling', type=int,
                        default=1, help="Unrolling which proceeds with each epoch")
    parser.add_argument('--nr_gt_steps', type=int,
                        default=2, help="Number of steps done by numerical solver")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')

    # Misc
    parser.add_argument('--print_interval', type=int, default=20,
            help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=False,
            help='pip the output to log file')

    # Add wandb arguments
    parser.add_argument('--wandb', type=bool, default=True,
                      help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='mp-pde-scnn',
                      help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='shtian-uc-san-diego',
                      help='WandB entity/username')

    args = parser.parse_args()
    
    # -----------------------------------------------
    # turn "250,100" → [250, 100], store both
    nt_base, nx_base = map(int, args.base_resolution.split(','))
    args.base_resolution = [nt_base, nx_base]  
    args.nt_base, args.nx_base = nt_base, nx_base

    main(args)
