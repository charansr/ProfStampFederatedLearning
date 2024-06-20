from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar, GetParametersIns, GetParametersRes, Status, Code, Parameters, FitIns, FitRes, \
    EvaluateRes, EvaluateIns

import torch
from torch.utils.data import Dataset, DataLoader
import flwr as fl

from model import MLP, train, test
from typing import List
from attacks import label_flipping_attack, targeted_label_flipping_attack
from attacks import controllable_mpaf_attack_nn
from dataset import get_data_numpy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import warnings
from logging import INFO
from flwr.common.logger import log
from omegaconf import DictConfig

import time


def generate_client_fn(traindataset_list: List[Dataset], valdataset_list: List[Dataset], num_classes: int, model: str,
                       cfg: DictConfig):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn_mlp(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClientMLP(
            traindataset=traindataset_list[int(cid)],
            valdataset=valdataset_list[int(cid)],
            num_classes=num_classes,
            weight_ratio=cfg["weight_attack_ratio"]
        ).to_client()


    # Control logic for other models
    # return the function to spawn client
    
    return client_fn_mlp

class FlowerClientMLP(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, traindataset: Dataset, valdataset: Dataset, num_classes: int, weight_ratio: float) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.traindataset = traindataset
        self.valdataset = valdataset

        # a model that is randomly initialised at first
        self.model = MLP(num_classes)
        self.num_classes = num_classes

        # figure out if this client has access to GPU support or not
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.attack_type = None
        self.is_malicious = False
        self.weight_ratio=weight_ratio

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        state_dict = self.model.state_dict()
        if self.attack_type == "MPAF" and self.is_malicious:
            state_dict = controllable_mpaf_attack_nn(state_dict, self.device, self.weight_ratio)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        # Poison the dataset if the client is malicious
        self.attack_type = config["attack_type"]
        self.is_malicious = config["is_malicious"]
        self.traindataset = applyAttacks(trainset=self.traindataset, config=config)

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        trainloader = DataLoader(
            self.traindataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
        train(self.model, trainloader, optim, epochs, self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        if config["defence"]:
            parameters = self.get_parameters({})
            # print("Testing--------------------------------------------------------------------------------")
            # time.sleep(60)
            loss, _, metrics = self.evaluate(parameters, config)
            metrics["loss"] = loss
            metrics["is_malicious"] = config["is_malicious"]
            return parameters, len(trainloader), metrics
        
        return self.get_parameters({}), len(trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        valloader = DataLoader(
            self.valdataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
        loss, accuracy, precision, recall, f1, conf_matrix = test(self.model, valloader, self.device)

        return float(loss), len(valloader), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                                             "confusion_matrix": conf_matrix}


def applyAttacks(trainset: Dataset, config, model: str = None) -> Dataset:
    # NOTE: this attack ratio is different, This is for number of samples to attack.
    ## The one in the config file is to select number of malicious clients

    if config["attack_type"] == "TLF":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return targeted_label_flipping_attack(trainset=trainset, attack_ratio=1.0)
    elif config["attack_type"] == "GAN":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return gan_attack(trainset=trainset)  # Change this if the program crashes
        # LGR model needs samples for all labels
        if model != "LGR":
            return partial_dataset_for_GAN_attack(trainset=trainset)
    elif config["attack_type"] == "MPAF":
        if config["is_malicious"]:
            print("----------------------------------Model Attacked------------------------------")
            return trainset
    else:
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return label_flipping_attack(dataset=trainset, num_classes=10, attack_ratio=1.0)

    return trainset