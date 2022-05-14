"""Task for deeplearning"""
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
import os
import torch
from torch import optim
from model import MinistClassfier
from dataset import MinistDataset


class MinistClassify:
    """Task for minist classify"""

    def __init__(self, config):
        device = torch.device(config.setup.device)
        data_cfg = config.data
        train_cfg = config.train
        self.model = MinistClassfier()

        self.model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), train_cfg.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.train_dataset = MinistDataset(
            "train",
            img_dir=os.path.join(data_cfg.data_root, "train", "images"),
            label_dir=os.path.join(data_cfg.data_root, "train", "labels_train.txt"),
            item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
            label_trans=lambda x: torch.tensor(x, dtype=torch.long),
        )
        self.valid_dataset = MinistDataset(
            "val",
            img_dir=os.path.join(data_cfg.data_root, "val", "images"),
            label_dir=os.path.join(data_cfg.data_root, "val", "labels_val.txt"),
            item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
            label_trans=lambda x: torch.tensor(x, dtype=torch.long),
        )
        self.pred_dataset = MinistDataset(
            "test",
            img_dir=os.path.join(data_cfg.data_root, "test", "images"),
            item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
        )
