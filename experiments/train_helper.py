# experiments/train_helper.py
# ---------------------------------------------------------------
import torch, random
from torch.utils.data import DataLoader
from torch import nn, optim
from common.utils import HDF5Dataset, GraphCreator
from equations.PDEs import *

# ---------------------------------------------------------------
def training_loop(model: nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: optim.Optimizer,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: nn.modules.loss._Loss,
                  device: torch.device = "cpu") -> torch.Tensor:
    """
    One epoch over `loader`; each sample is processed at a random
    starting time‑step and, optionally, un‑rolled (“push‑forward trick”).
    """

    losses = []
    is_graph = getattr(model, "is_graph_model", False)

    for (u_base, u_super, x, variables) in loader:
        optimizer.zero_grad()

        # ---- pick #unrollings & random start steps -----------------
        n_unroll = random.choice(unrolling)
        legal_steps = range(graph_creator.tw,
                            graph_creator.t_res - graph_creator.tw -
                            graph_creator.tw * n_unroll + 1)
        random_steps = random.choices(list(legal_steps), k=batch_size)

        data, labels = graph_creator.create_data(u_super, random_steps)

        if is_graph:
            graph = graph_creator.create_graph(
                data, labels, x, variables, random_steps
            ).to(device)
        else:
            data, labels = data.to(device), labels.to(device)

        # ---- push‑forward unrolling --------------------------------
        with torch.no_grad():
            for _ in range(n_unroll):
                random_steps = [s + graph_creator.tw for s in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)

                if is_graph:
                    pred = model(graph)
                    graph = graph_creator.create_next_graph(
                        graph, pred, labels, random_steps
                    ).to(device)
                else:
                    data = model(data)      # feed prediction back in
                    labels = labels.to(device)

        # ---- actual training step ---------------------------------
        if is_graph:
            pred = model(graph)
            loss = criterion(pred, graph.y)
        else:
            pred  = model(data)
            loss  = criterion(pred, labels)

        loss = torch.sqrt(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach() / batch_size)

    return torch.stack(losses)


# ---------------------------------------------------------------
def test_timestep_losses(model: nn.Module,
                         steps: list,
                         batch_size: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: nn.modules.loss._Loss,
                         device: torch.device = "cpu") -> None:
    """
    Evaluate single‑step prediction error at a list of time‑steps.
    """

    is_graph = getattr(model, "is_graph_model", False)

    for step in steps:
        if step != graph_creator.tw and step % graph_creator.tw != 0:
            continue

        losses = []
        for (u_base, u_super, x, variables) in loader:
            with torch.no_grad():
                same_steps = [step] * batch_size
                data, labels = graph_creator.create_data(u_super, same_steps)

                if is_graph:
                    graph = graph_creator.create_graph(
                        data, labels, x, variables, same_steps
                    ).to(device)
                    pred  = model(graph)
                    loss  = criterion(pred, graph.y)
                else:
                    data, labels = data.to(device), labels.to(device)
                    pred  = model(data)
                    loss  = criterion(pred, labels)

                losses.append(loss / batch_size)

        print(f"Step {step:4d} | mean loss {torch.mean(torch.stack(losses)):.4e}")


# ---------------------------------------------------------------
def test_unrolled_losses(model: nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: nn.modules.loss._Loss,
                         device: torch.device = "cpu") -> torch.Tensor:
    """
    Fully unroll the trajectory and accumulate the error.
    """

    is_graph = getattr(model, "is_graph_model", False)
    losses, losses_base = [], []

    for (u_base, u_super, x, variables) in loader:
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)

            if is_graph:
                graph = graph_creator.create_graph(
                    data, labels, x, variables, same_steps
                ).to(device)
                pred  = model(graph)
                loss  = criterion(pred, graph.y) / nx_base_resolution
            else:
                data, labels = data.to(device), labels.to(device)
                pred  = model(data)
                loss  = criterion(pred, labels) / nx_base_resolution

            batch_losses = [loss / batch_size]

            # un‑roll forward -------------------------------------------------
            for step in range(graph_creator.tw * (nr_gt_steps + 1),
                              graph_creator.t_res - graph_creator.tw + 1,
                              graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)

                if is_graph:
                    graph = graph_creator.create_next_graph(
                        graph, pred, labels, same_steps
                    ).to(device)
                    pred  = model(graph)
                    loss  = criterion(pred, graph.y) / nx_base_resolution
                else:
                    labels = labels.to(device)
                    pred   = model(pred)
                    loss   = criterion(pred, labels) / nx_base_resolution

                batch_losses.append(loss / batch_size)

            # numerical baseline ---------------------------------------------
            base_losses = []
            for step in range(graph_creator.tw * nr_gt_steps,
                              graph_creator.t_res - graph_creator.tw + 1,
                              graph_creator.tw):
                same_steps = [step] * batch_size
                _, y_super = graph_creator.create_data(u_super, same_steps)
                _, y_base  = graph_creator.create_data(u_base,  same_steps)
                base_losses.append(
                    criterion(y_super, y_base) / nx_base_resolution / batch_size
                )

        losses.append(torch.sum(torch.stack(batch_losses)))
        losses_base.append(torch.sum(torch.stack(base_losses)))

    print(f"Unrolled forward loss      : {torch.mean(torch.stack(losses)):.4e}")
    print(f"Unrolled numerical baseline: {torch.mean(torch.stack(losses_base)):.4e}")

    return torch.stack(losses)
