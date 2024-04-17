import os
import time
import logging
import logging.config
import numpy as np
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader  # Using PyG Data utilities
from torch.optim.lr_scheduler import LambdaLR
from ppo_controller import PPOController

logger = logging.getLogger(__name__)


def calculatedistance(
    decoded_points: list[list], alldistanceMatrix: list[np.array]
) -> list:
    rewards = torch.zeros((len(decoded_points), len(decoded_points[0]))).to("mps")
    completebatch = []
    for i, decoded_city in enumerate(decoded_points):
        # 0 to 0 will not matter as matrix is 0
        # depot is part of selection cities
        distanceMatrix = alldistanceMatrix[i]
        zero_decoded_city = torch.zeros((len(decoded_city) + 1)).long()

        zero_decoded_city[1:] = decoded_city
        row_edge_distances = distanceMatrix[
            zero_decoded_city[:-1], zero_decoded_city[1:]
        ]
        completebatch.append(row_edge_distances)
    batch_rewards = torch.stack(completebatch, axis=1)
    # reward is 0 at the start
    rewards = batch_rewards.permute(1, 0)
    return rewards


# Boiler Plate Code From BD4H and DL Class for recording metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
    forward_network,
    critic,
    device,
    data_loader,
    valid_loader,
    criterion,
    epoch,
    batch_size,
    customercount,
    ppo_epochs,
    update_policies_at,
    num_decoding_steps,
    print_freq=50,
):
    controller = PPOController(ppo_epochs, update_policies_at, forward_network, critic)
    memory = controller.memory
    for i, data in enumerate(data_loader):
        import pdb

        pdb.set_trace()

        # convert to batch for storing it in memory
        # let it remain in RAM else it would have to be pushed back
        # from GPU to CPU
        edge_attr = data.edge_attr.view(batch_size, customercount * (customercount - 1))
        demand = data.y.view(batch_size, customercount)

        # push to GPU for training
        data = data.to(device)
        # Prepare training
        controller.forward_network_old.train()
        controller.optimizer_forward_network_old.zero_grad()
        controller.optimizer_critic_old.zero_grad()
        # Pass the batch through the network
        # this can be random/greedy and a flag needs to be passed

        actions, log_p, entropy, dists, x = controller.forward_network_old(
            data, 0, num_decoding_steps, batch_size, False, False
        )
        rewards = calculatedistance(
            decoded_points=actions, alldistanceMatrix=data.distance_matrix
        )
        # Putting it back to cpu to save it to RAM instead of GPU memory
        actions = actions.to(torch.device("cpu")).detach()
        log_p = log_p.to(torch.device("cpu")).detach()
        rewards = rewards.to(torch.device("cpu")).detach()
        for i_batch in range(batch_size):
            memory.input_x.append(x[i_batch])
            memory.input_attr.append(edge_attr[i_batch])
            memory.actions.append(actions[i_batch])
            memory.log_probs.append(log_p[i_batch])
            memory.rewards.append(rewards[i_batch])
            memory.capacity.append(torch.tensor(forward_network_old.vehicle_capacity))
            memory.demand.append(demand[i_batch])
        if (i + 1) % update_policies_at == 0:
            # Updating the new policies for both actor and critic using memory samples
            controller.update_policies(memory, epoch)
            # Clearing memory - fill memory with old policy before using
            memory.clear_memory()
        costs = []
        cost = rollout(
            controller.optimizer_forward_network,
            valid_loader,
            batch_size,
            num_decoding_steps,
        )
        cost = cost.mean()
        costs.append(cost.item())
        print("Problem:TSP" "%s" % n_nodes, "/ Average distance:", cost.item())
        print(costs)


def rollout(forward_network, dataset, batch_size, steps):
    forward_network.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            actions, log_p, entropy, dists, x = forward_network(
                bat, 0, steps, batch_size, True, False
            )

            rewards = calculatedistance(
                decoded_points=actions, alldistanceMatrix=data.distance_matrix
            )
        return rewards.cpu()

    totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return totall_cost
