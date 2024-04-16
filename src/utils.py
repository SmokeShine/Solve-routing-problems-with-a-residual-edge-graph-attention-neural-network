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
    criterion,
    optimizer_forward_network,
    optimizer_critic,
    epochs,
    batch_size,
    customercount,
    print_freq=50,
):
    # Create a copy for PPO
    forward_network_old = copy.deepcopy(forward_network)
    critic_old = copy.deepcopy(critic)
    # should not be required. check and delete later
    forward_network_old.load_state_dict(forward_network.state_dict())
    critic_old.load_state_dict(critic.state_dict())
    # Memory for sampling
    memory = Memory()
    # Push to device
    forward_network_old.to(device)
    critic_old.to(device)
    forward_network.to(device)
    critic.to(device)

    forward_network_old.train()
    critic_old.train()

    losses_forward_network = AverageMeter()
    losses_critic = AverageMeter()
    total_losses = AverageMeter()
    for epoch in range(epochs):
        for i, data in enumerate(data_loader):
            import pdb

            pdb.set_trace()

            # convert to batch for storing it in memory
            # let it remain in RAM else it would have to be pushed back
            # from GPU to CPU
            edge_attr = data.edge_attr.view(
                batch_size, customercount * (customercount - 1)
            )
            demand = data.y.view(batch_size, customercount)

            # push to GPU for training
            data = data.to(device)
            # Prepare training
            optimizer_forward_network.zero_grad()
            optimizer_critic.zero_grad()
            # Pass the batch through the network
            actions, log_p, entropy, dists, x = forward_network_old(data)
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
                memory.capacity.append(
                    torch.tensor(forward_network_old.vehicle_capacity)
                )
                memory.demand.append(demand[i_batch])
            if (i + 1) % 4 == 0:
                # Updating the new policy using memory samples
                for i in range(old_input_x.size(0)):
                    data = Data(
                        x=old_input_x[i],
                        edge_index=edges_index,
                        edge_attr=old_input_attr[i],
                        actions=old_action[i],
                        rewards=old_rewards[i],
                        log_probs=old_log_probs[i],
                        demand=old_demand[i],
                        capcity=old_capcity[i],
                    )
                    datas.append(data)

                memory.clear_memory()

            output_critic = critic(
                location_embedding=static_location_embedding,
                log_probabilities=output_actor_prob,
            )

            # target
            rewards = calculatedistance(
                decoded_points=output_actor_points, alldistanceMatrix=distanceMatrix
            )
            advantage = rewards - output_critic
            pointer_reward = torch.max(output_actor_prob, axis=2)
            # actor loss
            # https://stackoverflow.com/questions/65815598/calling-backward-function-for-two-different-neural-networks-but-getting-retai
            loss_actor = torch.mean(torch.sum(advantage * pointer_reward[0], axis=1))

            # critic loss
            loss_critic = torch.mean(torch.sum(torch.square(advantage), axis=1))
            assert not np.isnan(loss_actor.item()), "Actor diverged with loss = NaN"
            assert not np.isnan(loss_critic.item()), "Critic diverged with loss = NaN"

            # Calculating total loss
            total_loss = loss_actor + loss_actor
            # calculating loss for end leaves of graph for both actor and critic
            total_loss.backward()
            # Clipping
            nn.utils.clip_grad_norm_(
                forward_network.parameters(), max_norm=2.0, norm_type=2
            )
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2.0, norm_type=2)
            # propogate loss back to critic graph
            optimizer_critic.step()
            # propogate loss back to actor graph
            optimizer_forward_network.step()

            # Updating Averagemeter
            losses_forward_network.update(loss_actor.item(), output_critic.size(0))
            losses_critic.update(loss_critic.item(), output_critic.size(0))
            if i % print_freq == 0:
                logger.info(
                    f"Epoch: {epoch} \t iteration: {i} \t Training Loss Actor Current:{losses_forward_network.val:.4f} Average:({losses_forward_network.avg:.4f})"
                )
                logger.info(
                    f"Epoch: {epoch} \t iteration: {i} \t Training Loss Critic Current:{losses_critic.val:.4f} Average:({losses_critic.avg:.4f})"
                )

    return [losses_forward_network.avg, losses_critic.avg]


def evaluate(model, device, data_loader, criterion, optimizer, print_freq=10):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), target.size(0))
            if i % print_freq == 0:
                logger.info(
                    f"Validation Loss Current:{losses.val:.4f} Average:({losses.avg:.4f})"
                )
    return losses.avg


def save_checkpoint(model, optimizer, path):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, path)
    torch.save(model, "./checkpoint_model.pth", _use_new_zipfile_serialization=False)
    logger.info(f"checkpoint saved at {path}")


class Memory:
    def __init__(self):
        self.input_x = []
        # self.input_index = []
        self.input_attr = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.capacity = []
        self.demand = []

    def clear_memory(self):
        self.input_x.clear()
        # self.input_index.clear()
        self.input_attr.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.capacity.clear()
        self.demand.clear()


class Actor_critic(nn.Module):
    def __init__(
        self,
        input_node_dim,
        hidden_node_dim,
        input_edge_dim,
        hidden_edge_dim,
        conv_laysers,
    ):
        super(Actor_critic, self).__init__()
        self.actor = ResidualEGAT()
        self.critic = Critic()

    def act(self, datas, actions, steps, batch_size, greedy, _action):
        actions, log_p, _, _, _ = self.actor(
            datas, actions, steps, batch_size, greedy, _action
        )

        return actions, log_p

    def evaluate(self, datas, actions, steps, batch_size, greedy, _action):
        _, _, entropy, old_log_p, x = self.actor(
            datas, actions, steps, batch_size, greedy, _action
        )

        value = self.critic(x)

        return entropy, old_log_p, value
