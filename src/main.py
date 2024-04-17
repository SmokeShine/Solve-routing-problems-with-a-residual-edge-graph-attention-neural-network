#!/usr/bin/env python

import logging
import logging.config
import numpy as np

# from plots import plot_loss_curve
from scipy.spatial.distance import pdist, squareform
import warnings
import datetime
import argparse
from utils import train, evaluate, save_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim


# from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader  # Using PyG Data utilities
import random
import mymodels
import numpy as np
from tqdm import tqdm

# warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

torch.manual_seed(42)

try:
    logging.config.fileConfig(
        "logging.ini",
        disable_existing_loggers=False,
        defaults={
            "logfilename": datetime.datetime.now().strftime(
                "../logs/ResidualEGAT_%H_%M_%d_%m.log"
            )
        },
    )
except:
    pass

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)


# represent the data as graphs.
# represent each instance as a graph where nodes represent locations,
# and edges represent distances between locations.


class SequenceWithLabelDataset(Dataset):
    def __init__(
        self,
        input_cities: list[int] = [],
        ninstances: int = None,
        max_demand: int = None,
    ):
        self.input_cities = input_cities
        self.ninstances = ninstances
        self.max_demand = max_demand
        self.data_list = self.create_data()
        if len(self.data_list) == 0:
            raise ValueError("No Data Generated")

    def create_data(self) -> list[Data]:
        data_list = []
        edge_index = self.create_edges(
            self.input_cities
        )  # needs to be called only once as this is a dense graph
        for _ in tqdm(range(self.ninstances)):
            locations, depot = self.create_instance(city=self.input_cities)
            distance_matrix = self.distance_matrix(locations)

            demand = np.random.randint(
                low=2, high=self.max_demand, size=self.input_cities
            )

            demand[0] = 0  # For depot

            x = torch.FloatTensor(locations)

            # remove diagonal elements from distance matrix
            edges = []
            for i in range(len(distance_matrix)):
                for j in range(len(distance_matrix)):
                    if i != j:
                        edges.append(distance_matrix[i][j])
            edge_attr = torch.FloatTensor(edges)
            y = torch.FloatTensor(demand)
            depot = torch.tensor(depot)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                depot=depot,
                distance_matrix=distance_matrix,
            )
            data_list.append(data)
        return data_list

    def create_instance(
        self, city: int
    ) -> tuple[list[tuple[float, float]], tuple[float, float]]:
        locations = [(random.random(), random.random()) for _ in range(city)]
        depot = locations[0]
        return locations, depot

    def distance_matrix(self, locations: list[tuple[float, float]]) -> np.ndarray:
        distvec = pdist(locations)
        distvec = np.float32(distvec)
        allpairs = squareform(distvec)
        return allpairs

    def create_edges(self, num_nodes: int) -> torch.LongTensor:
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Data:
        return self.data_list[index]


def train_model(model_name="ResidualEGAT"):
    logger.info("Generating DataLoader")
    logger.info("Training Model")
    forward_network = mymodels.ResidualEGAT(
        vehicle_capacity=MAXDEMAND
    )  # homogenous VRP
    critic = mymodels.Critic()
    logger.info(f"Forward Network: {forward_network}")
    logger.info(f"Critic: {critic}")
    save_file = "ResidualEGAT.pth"

    train_dataset = SequenceWithLabelDataset(
        input_cities=CUSTOMERCOUNT,
        ninstances=10,  # 1 million instances,
        max_demand=MAXDEMAND,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    valid_dataset = SequenceWithLabelDataset(
        input_cities=CUSTOMERCOUNT,
        ninstances=5,  # variable length batches
        max_demand=MAXDEMAND,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    criterion = nn.MSELoss()
    # torch.sqrt in the training loop
    criterion.to(DEVICE)
    train_actor_loss_history = []
    train_critic_loss_history = []
    valid_loss_history = []
    best_validation_loss = float("inf")
    early_stopping_counter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}")

        train_loss = train(
            forward_network,
            critic,
            DEVICE,
            train_loader,
            valid_loader,
            criterion,
            epoch,
            BATCH_SIZE,
            CUSTOMERCOUNT,
            PPO_EPOCH,
            UPDATE_POLICIES_AT,
            NUM_DECODING_STEPS,
        )
        logger.info(
            f"Forward Network: Average Loss for epoch {epoch} is {train_loss[0]}"
        )
        logger.info(f"Critic: Average Loss for epoch {epoch} is {train_loss[1]}")
        train_actor_loss_history.append(train_loss[0])
        train_critic_loss_history.append(train_loss[1])

    #     valid_loss = evaluate(actor, DEVICE, valid_loader, criterion, optimizer)
    #     valid_loss_history.append(valid_loss)
    #     is_best = best_validation_loss > valid_loss
    #     if epoch % EPOCH_SAVE_CHECKPOINT == 0:
    #         logger.info(f"Saving Checkpoint for {model_name} at epoch {epoch}")
    #         save_checkpoint(model, optimizer, save_file + "_" + str(epoch) + ".tar")
    #     if is_best:
    #         early_stopping_counter = 0
    #         logger.info(
    #             f"New Best Identified: \t Old Loss: {best_validation_loss}  vs New loss:\t{valid_loss} "
    #         )
    #         best_validation_loss = valid_loss
    #         torch.save(model, "./best_model.pth", _use_new_zipfile_serialization=False)
    #     else:
    #         logger.info("Loss didnot improve")
    #         early_stopping_counter += 1
    #     if early_stopping_counter >= PATIENCE:
    #         break
    # # final checkpoint saved
    # save_checkpoint(actor, optimizer, save_file + ".tar")
    # # Loading Best Model
    # best_model = torch.load("./checkpoint_model.pth")
    # logger.info(f"Train Losses:{train_loss_history}")
    # logger.info(f"Validation Losses:{valid_loss_history}")
    # logger.info(f"Plotting Charts")

    # plot_loss_curve(
    #     model_name,
    #     train_loss_history,
    #     valid_loss_history,
    #     "Loss Curve",
    #     f"{PLOT_OUTPUT_PATH}loss_curves.jpg",
    # )
    # logger.info(f"Train Losses:{train_loss_history}")
    logger.info(f"Training Finished for {model_name}")


def predict_model(best_model):
    pred_dataset = SequenceWithLabelDataset(input_cities=[CUSTOMERCOUNT], ninstances=1)
    pred_loader = DataLoader(
        dataset=pred_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    # No need to calculate grad as it is forward pass only
    best_model.eval()
    with torch.no_grad():
        for counter, (input, target) in tqdm(enumerate(pred_loader)):
            # Model is in GPU
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            output = best_model(input)
            logger.info(f"Input:{input}")
            _, pointer = torch.max(output, axis=1)
            logger.info(f"Prediction:{pointer}")
            logger.info(f"Target:{target}")
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Residual E-GAT for VRP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--customers",
        default=15,
        type=int,
        choices=range(1, 100),
        help="Number of customers",
    )
    parser.add_argument(
        "--maxdemand",
        default=15,
        type=int,
        choices=range(1, 100),
        help="Maximum Demand allowed for a customer",
    )
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train Model"
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=128,
        help="Batch size for training the model",
    )
    parser.add_argument(
        "--num_workers", nargs="?", type=int, default=0, help="Number of Available CPUs"
    )
    parser.add_argument(
        "--num_epochs",
        nargs="?",
        type=int,
        default=100,
        help="Number of Epochs for training the model",
    )
    parser.add_argument(
        "--ppo_epoch",
        nargs="?",
        type=int,
        default=10,
        help="Number of Epochs for PPO",
    )
    parser.add_argument(
        "--update_policies_at",
        nargs="?",
        type=int,
        default=4,
        help="Use samples from memory to update the policies",
    )
    parser.add_argument(
        "--num_decoding_steps",
        nargs="?",
        type=int,
        default=1000,
        help="Use samples from memory to update the policies",
    )
    parser.add_argument(
        "--learning_rate",
        nargs="?",
        type=float,
        default=0.0003,
        help="Learning Rate for the optimizer",
    )

    parser.add_argument(
        "--plot_output_path", default="./Plots_", help="Output path for Plot"
    )
    parser.add_argument("--model_path", help="Model Path to resume training")
    parser.add_argument(
        "--epoch_save_checkpoint",
        nargs="?",
        type=int,
        default=5,
        help="Epochs after which to save model checkpoint",
    )
    parser.add_argument(
        "--pred_model",
        default="./checkpoint_model.pth",
        help="Model for prediction; Default is checkpoint_model.pth; \
                            change to ./best_model.pth for 1 sample best model",
    )
    parser.add_argument(
        "--patience", nargs="?", type=int, default=5, help="Early stopping epoch count"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    CUSTOMERCOUNT = args.customers

    global BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, LEARNING_RATE, PRED_MODEL, PATIENCE, MAXDEMAND, PPO_EPOCH, UPDATE_POLCIIES_AT, NUM_DECODING_STEPS
    MAXDEMAND = args.maxdemand
    PPO_EPOCH = args.ppo_epoch
    __train__ = args.train
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    UPDATE_POLICIES_AT = args.update_policies_at
    PLOT_OUTPUT_PATH = args.plot_output_path
    EPOCH_SAVE_CHECKPOINT = args.epoch_save_checkpoint
    MODEL_PATH = args.model_path
    PATIENCE = args.patience
    PRED_MODEL = args.pred_model
    DEVICE = torch.device("mps")
    NUM_DECODING_STEPS = args.num_decoding_steps

    logger.info(f"Problem Size:{CUSTOMERCOUNT}")

    if __train__:
        logger.info("Training")
        train_model()
    else:
        logger.info("Prediction")
        best_model = torch.load(PRED_MODEL)
        logger.info(f"Using {PRED_MODEL} for prediction")
        predict_model(best_model)
        logger.info("Prediction Step Complete")
