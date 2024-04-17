
# Solve routing problems with a residual edge-graph attention neural network
This repository contains pytorch implementation for solving VRP using Reinforcement Learning[1]

# Details
For each epoch, ppo controller keeps two copies of actor and critic. One is used to explore while the other copy is updated after interval using PPO samples which allows stability in the training.  

## Actor
### Encoder
Single headed GAT modified to include residual connection. Input is location and edge vector. All the aggregated embeddings via GNN message passing are then added to create a context vector.

### Decoder
The context vector along with the input nodes is used to create a pointer for the next destination. Multi headed transformer is used which created multiple vectors. Then it is passed to another attention layer which gives a probability score. The node with the maximum probability can be chosen if greedy search is selected.

## Critic
Takes input of actor and tries to predict the reward.


# Links
[1] Kun Lei, Peng Guo, Yi Wang, Xiao Wu, Wenchao Zhao. Solve routing problems with a residual edge-graph attention neural network. Neurocomputing.2022.
<https://arxiv.org/abs/2105.02730>
Rewrite for the repo -  <https://github.com/pengguo318/RoutingProblemGANN>
