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


logger = logging.getLogger(__name__)


class PPOController(object):
    # Main controller
    # For each epoch, initiates training of actor/critic network
    # observes and saves samples
    # update both actor/critic network using PPO
    # to avoid instability

    def __init__(
        self,
        ppo_epochs: int,
        update_networks_at,
        forward_network,
        critic,
    ) -> None:
        self.ppo_epochs = ppo_epochs
        self.update_networks_at = update_networks_at
        self.memory = Memory()
        self.forward_network = forward_network
        self.critic = critic
        self.create_networks()

    def create_networks(self):
        self.forward_network_old = copy.deepcopy(self.forward_network)
        self.critic_old = copy.deepcopy(self.critic)
        # should not be required. check and delete later
        self.forward_network_old.load_state_dict(self.forward_network.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        self.optimizer_forward_network = torch.optim.Adam(
            self.forward_network.parameters(), lr=3e-4
        )
        self.optimizer_forward_network_old = torch.optim.Adam(
            self.forward_network_old.parameters(), lr=3e-4
        )
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.optimizer_critic_old = torch.optim.Adam(
            self.critic_old.parameters(), lr=3e-4
        )

    def update_policies(self, memory, epoch):
        old_input_x = torch.stack(memory.input_x)
        # old_input_index = torch.stack(memory.input_index)
        old_input_attr = torch.stack(memory.input_attr)
        old_demand = torch.stack(memory.demand)
        old_capcity = torch.stack(memory.capacity)

        old_action = torch.stack(memory.actions)
        old_rewards = torch.stack(memory.rewards).unsqueeze(-1)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(-1)

        datas = []
        edges_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                edges_index.append([i, j])
        edges_index = torch.LongTensor(edges_index)
        edges_index = edges_index.transpose(dim0=0, dim1=1)
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
        # print(np.array(datas).shape)
        self.policy.to(device)
        data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=False)
        # 学习率退火
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda f: 0.96**epoch)
        value_buffer = 0

        for i in range(self.epoch):

            self.policy.train()
            epoch_start = time.time()
            start = epoch_start
            self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

            for batch_idx, batch in enumerate(data_loader):
                self.batch_idx += 1
                batch = batch.to(device)
                entropy, log_probs, value = self.policy.evaluate(
                    batch,
                    batch.actions,
                    self.steps,
                    self.batch_size,
                    self.greedy,
                    self._action,
                )
                # advangtage function

                # base_reward = self.adv_normalize(base_reward)
                rewar = batch.rewards
                rewar = self.adv_normalize(rewar)
                # rewar = rewar/torch.max(rewar)
                # Value function clipping
                mse_loss = self.MseLoss(rewar, value)

                ratios = torch.exp(log_probs - batch.log_probs)

                # norm advantages
                advantages = rewar - value.detach()

                # advantages = self.adv_normalize(advantages)
                # PPO loss
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )
                # total loss
                loss = (
                    torch.min(surr1, surr2)
                    + 0.5 * mse_loss
                    - self.entropy_value * entropy
                )
                self.optimizer_forward_network.zero_grad()
                self.optimizer_critic.zero_grad()

                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(
                    self.forward_network.parameters(), max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.optimizer_forward_network.step()
                self.optimizer_critic.step()
                scheduler.step()

                self.rewards.append(torch.mean(rewar.detach()).item())
                self.losses.append(torch.mean(loss.detach()).item())
                # print(epoch,self.optimizer.param_groups[0]['lr'])

        self.forward_network_old.load_state_dict(self.forward_network.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())


def update_state(demand, dynamic_capcity, selected, c=20):  # dynamic_capcity(num,1)

    depot = selected.squeeze(-1).eq(0)  # Is there a group to access the depot

    current_demand = torch.gather(demand, 1, selected)

    dynamic_capcity = dynamic_capcity - current_demand
    if depot.any():
        dynamic_capcity[depot.nonzero().squeeze()] = c

    return dynamic_capcity.detach()  # (bach_size,1)


def update_mask(demand, capcity, selected, mask, i):
    go_depot = selected.squeeze(-1).eq(
        0
    )  # If there is a route to select a depot, mask the depot, otherwise it will not mask the depot
    # print(go_depot.nonzero().squeeze())
    # visit = selected.ne(0)

    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    if (~go_depot).any():
        mask1[(~go_depot).nonzero(), 0] = 0

    if i + 1 > demand.size(1):
        is_done = (mask1[:, 1:].sum(1) >= (demand.size(1) - 1)).float()
        combined = is_done.gt(0)
        mask1[combined.nonzero(), 0] = 0
        """for i in range(demand.size(0)):
            if not mask1[i,1:].eq(0).any():
                mask1[i,0] = 0"""
    a = demand > capcity
    mask = a + mask1

    return mask.detach(), mask1.detach()


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
