import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR
import time
import math


class ResidualEGAT(nn.Module):
    def __init__(self, vehicle_capacity: int) -> None:
        super(ResidualEGAT, self).__init__()
        self.vehicle_capacity = vehicle_capacity
        self.actor = Actor(
            vehicle_capacity=self.vehicle_capacity
        )  # to be passed to decoder

    def forward(self, data):
        actions, log_p, entropy, dists, context_vector = self.actor(data)
        return actions, log_p, entropy, dists, context_vector


# an actor network that predicts a probability distribution over the next action at any given decision step,
# Embeddings for edge and (x,demand,load-demand) here all three are combined considered as x
# attention is part of encoding as well as decoding - single head in encoding and multi head in decoding
# attention is used to create residual between GNN layers.
# how to create gnn layers and how to residual between them
# what is the output of actor as this will be used for calculating loss
# average will create the context vector
# context will go through multi head to create the next pointer
# proximal policy - let it be same for the time being
class Actor(nn.Module):
    # Consists of encoder and decoder
    def __init__(self, vehicle_capacity: int) -> None:
        super(Actor, self).__init__()
        self.vehicle_capacity = vehicle_capacity
        # Encoder will use the batch of instances to create a disconnected graph with all nodes
        # Message Passing will work
        self.encoder: Encoder = Encoder()
        # Create a context vector using mean of all embeddings of nodes 1x1
        self.decoder: DecoderTransformer = DecoderTransformer(
            vehicle_capacity=self.vehicle_capacity
        )  # Use vehicle capacity to mask nodes

    def forward(
        self, data, actions_old, _action, n_steps=1, batch_size=512, greedy=False
    ):
        embedding_w_residual = self.encoder(data)
        context_vector = embedding_w_residual.mean(dim=1)  # Create a context vector 1x1
        demand = data.y
        capacity = self.vehicle_capacity

        actions, log_p, entropy, dists = self.decoder(
            data,
            context_vector,
            actions_old,
            capacity,
            demand,
            n_steps,
            batch_size,
            greedy,
            _action,
        )
        return actions, log_p, entropy, dists, data


class Encoder(nn.Module):
    def __init__(
        self,
        input_node_dim=3,
        hidden_node_dim=128,
        input_edge_dim=1,
        hidden_edge_dim=16,
        conv_layers=3,
    ):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(input_node_dim, hidden_node_dim)
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.be = nn.BatchNorm1d(hidden_edge_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)  # 1-16
        self.convs1 = nn.ModuleList(
            [
                GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim)
                for _ in range(conv_layers)
            ]
        )

    def forward(self, data):
        batch_size = data.num_graphs
        # print(batch_size)
        # edge_attr = data.edge_attr

        x = torch.cat([data.x, data.y], -1)
        x = self.fc_node(x)
        x = self.bn(x)
        edge_attr = self.fc_edge(data.edge_attr)
        edge_attr = self.be(edge_attr)
        for conv in self.convs1:
            # x = conv(x,data.edge_index)
            x1 = conv(x, data.edge_index, edge_attr)
            x = x + x1
        # Create an embedding for each node in the disconnected graph
        x = x.reshape(
            (batch_size, -1, self.hidden_node_dim)
        )  # Required for connecting to normal pytorch layers

        return x


class GatConv(MessagePassing):
    def __init__(
        self, in_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0
    ):
        super(GatConv, self).__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels + edge_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = nn.LeakyReLU(alpha, self.negative_slope)
        alpha = torch.softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = torch.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


# Change initialization of layers - paper publication


# multi head attention, followed by attention
# only decoder requires attention.
# encoder attention is single head and
# can be done in message passing
class Attention(nn.Module):
    def __init__(
        self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0
    ):
        super(Attention, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, state_t, context, mask):
        """
        :param state_t: (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        :param context: （batch_size,n_nodes,input_dim）
        :param mask: selected nodes  (batch_size,n_nodes)
        :return:
        """
        batch_size, n_nodes, input_dim = context.size()
        Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        compatibility = self.norm * torch.matmul(
            Q, K.transpose(2, 3)
        )  # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        compatibility = compatibility.squeeze(2)  # (batch_size,n_heads,n_nodes)
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))

        scores = F.softmax(u_i, dim=-1)  # (batch_size,n_heads,n_nodes)
        scores = scores.unsqueeze(2)
        out_put = torch.matmul(
            scores, V
        )  # (batch_size,n_heads,1,n_nodes )*(batch_size,n_heads,n_nodes,head_dim)
        out_put = out_put.squeeze(2).view(
            batch_size, self.hidden_dim
        )  # （batch_size,n_heads,hidden_dim）
        out_put = self.fc(out_put)

        return out_put  # (batch_size,hidden_dim)


class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = Attention(n_heads, 1, input_dim, hidden_dim)

    def forward(self, state_t, context, mask):
        """
        :param state_t: (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        :param context: （batch_size,n_nodes,input_dim）
        :param mask: selected nodes  (batch_size,n_nodes)
        :return:softmax_score
        """
        x = self.mhalayer(state_t, context, mask)

        batch_size, n_nodes, input_dim = context.size()
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = self.norm * torch.matmul(
            Q, K.transpose(1, 2)
        )  # (batch_size,1,n_nodes)
        compatibility = compatibility.squeeze(1)
        x = torch.tanh(compatibility)
        x = x * (10)
        x = x.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(x, dim=-1)
        return scores


# Masking depot if previous is depot


# decoder is not a transformer in official implementation. it is ptrnet.
# multi head probability, again attention to get a single max value
class DecoderTransformer(nn.Module):
    def __init__(self, vehicle_capacity, input_dim=128, hidden_dim=128):
        super(DecoderTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prob = ProbAttention(8, input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        encoder_inputs,
        pool,
        actions_old,
        capcity,
        demand,
        n_steps,
        batch_size,
        greedy=False,
        _action=False,
    ):

        mask1 = encoder_inputs.new_zeros(
            (encoder_inputs.size(0), encoder_inputs.size(1))
        )
        mask = encoder_inputs.new_zeros(
            (encoder_inputs.size(0), encoder_inputs.size(1))
        )
        # 用old_policy来sample的action
        dynamic_capcity = capcity.view(encoder_inputs.size(0), -1)  # bat_size
        demands = demand.view(
            encoder_inputs.size(0), encoder_inputs.size(1)
        )  # （batch_size,seq_len）
        index = torch.zeros(encoder_inputs.size(0)).to(device).long()
        if _action:
            actions_old = actions_old.reshape(batch_size, -1)
            entropys = []
            old_actions_probs = []

            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]  # depot

                # -----------------------------------------------------------------------------GRU做信息传递
                # hx = self.cell(_input, hx)
                # decoder_input = hx
                # -----------------------------------------------------------------------------pool+cat(first_node,current_node)
                decoder_input = torch.cat([_input, dynamic_capcity], -1)
                decoder_input = self.fc(decoder_input)
                pool = self.fc1(pool)
                decoder_input = decoder_input + pool

                # -----------------------------------------------------------------------------cat(pool,first_node,current_node)
                """decoder_input = torch.cat([pool, _input, dynamic_capcity], dim=-1)
                decoder_input = self.fc(decoder_input)"""

                if i == 0:
                    mask, mask1 = update_mask(
                        demands, dynamic_capcity, index.unsqueeze(-1), mask1, i
                    )

                # decoder_input = torch.cat([pool,_input_first], dim=-1)
                # decoder_input  = self.fc(decoder_input)
                # -----------------------------------------------------------------------------------------------------------
                p = self.prob(decoder_input, encoder_inputs, mask)

                dist = Categorical(p)

                old_actions_prob = dist.log_prob(actions_old[:, i])
                entropy = dist.entropy()
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                old_actions_prob = old_actions_prob * (1.0 - is_done)
                entropy = entropy * (1.0 - is_done)

                entropys.append(entropy.unsqueeze(1))
                old_actions_probs.append(old_actions_prob.unsqueeze(1))

                dynamic_capcity = update_state(
                    demands,
                    dynamic_capcity,
                    actions_old[:, i].unsqueeze(-1),
                    capcity[0].item(),
                )
                mask, mask1 = update_mask(
                    demands, dynamic_capcity, actions_old[:, i].unsqueeze(-1), mask1, i
                )

                _input = torch.gather(
                    encoder_inputs,
                    1,
                    actions_old[:, i]
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(encoder_inputs.size(0), -1, encoder_inputs.size(2)),
                ).squeeze(1)

            # log_ps = torch.cat(log_ps,dim=1)
            # actions = torch.cat(actions,dim=1)
            entropys = torch.cat(entropys, dim=1)
            old_actions_probs = torch.cat(old_actions_probs, dim=1)
            # log_p = log_ps.sum(dim=1)
            num_e = entropys.ne(0).float().sum(1)
            entropy = entropys.sum(1) / num_e
            old_actions_probs = old_actions_probs.sum(dim=1)

            return 0, 0, entropy, old_actions_probs
        else:
            log_ps = []
            actions = []
            # entropys = []
            # h0 = self.h0.unsqueeze(0).expand(encoder_inputs.size(0), -1)
            # first_cat = self._input[None, :].expand(encoder_inputs.size(0), -1)
            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]  # depot
                # -----------------------------------------------------------------------------GRU做信息传递
                # hx = self.cell(_input, hx)
                # decoder_input = hx
                # -----------------------------------------------------------------------------pool+cat(first_node,current_node)
                decoder_input = torch.cat([_input, dynamic_capcity], -1)
                decoder_input = self.fc(decoder_input)
                pool = self.fc1(pool)
                decoder_input = decoder_input + pool
                # -----------------------------------------------------------------------------cat(pool,first_node,current_node)
                """decoder_input = torch.cat([pool,_input,dynamic_capcity], dim=-1)
                decoder_input  = self.fc(decoder_input)"""
                # -----------------------------------------------------------------------------------------------------------
                if i == 0:
                    mask, mask1 = update_mask(
                        demands, dynamic_capcity, index.unsqueeze(-1), mask1, i
                    )
                p = self.prob(decoder_input, encoder_inputs, mask)
                dist = Categorical(p)
                if greedy:
                    _, index = p.max(dim=-1)
                else:
                    index = dist.sample()

                actions.append(index.data.unsqueeze(1))
                log_p = dist.log_prob(index)
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                log_p = log_p * (1.0 - is_done)

                log_ps.append(log_p.unsqueeze(1))

                # entropys.append(entropy.unsqueeze(1))
                dynamic_capcity = update_state(
                    demands, dynamic_capcity, index.unsqueeze(-1), capcity[0].item()
                )
                mask, mask1 = update_mask(
                    demands, dynamic_capcity, index.unsqueeze(-1), mask1, i
                )

                _input = torch.gather(
                    encoder_inputs,
                    1,
                    index.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(encoder_inputs.size(0), -1, encoder_inputs.size(2)),
                ).squeeze(1)
            log_ps = torch.cat(log_ps, dim=1)
            actions = torch.cat(actions, dim=1)

            log_p = log_ps.sum(dim=1)

            return actions, log_p, 0, 0


# a critic network that estimates the reward for any problem instance from a given state
class Critic(nn.Module):
    # output probabilities of actor to compute weighted sum of embedded inputs
    def __init__(self, hidden_node_dim=128) -> None:
        """Estimates the problem complexity.

        This is a basic module that just looks at the log-probabilities predicted by
        the encoder + decoder, and returns an estimate of complexity
        """

        super(Critic, self).__init__()

        self.fc1 = nn.Conv1d(hidden_node_dim, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

    def forward(self, x):
        x1 = x.transpose(2, 1)
        output = torch.relu(self.fc1(x1))
        output = torch.relu(self.fc2(output))
        value = self.fc3(output).sum(dim=2).squeeze(-1)
        return value
