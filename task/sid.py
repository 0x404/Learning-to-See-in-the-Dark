"Task of see in dark (SID)"
from pathlib import Path
import torch
from torch import optim
from model import UNet
from dataset import DataSetSID
from dataset.sid import filp_transform, rotation_transform, snoy_pack


class SeeInDark:
    def __init__(self, config):
        model_cfg = config.model
        data_cfg = config.data
        train_cfg = config.train
        device = torch.device(config.setup.device)

        # mode config
        self.model = UNet(model_cfg.channel_in, model_cfg.channel_out)
        self.model.to(device=device)

        # loss function
        self.loss_function = torch.nn.L1Loss()
        self.loss_function.to(device=device)

        # # optimizer
        # self.optimizer = optim.Adam(self.model.parameters, lr=train_cfg.lr)

        # lr scheduler
        # self.lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer=self.optimizer,
        #     step_size=train_cfg.lr_step,
        #     gamma=train_cfg.lr_gamma,
        # )

        # dataset conifg
        transfrom = []
        if data_cfg.transform.use_flip:
            transfrom.append(filp_transform)
        if data_cfg.transform.use_rotation:
            transfrom.append(rotation_transform)
        self.train_dataset = DataSetSID(
            config=data_cfg,
            file=Path(data_cfg.data_root).joinpath("Sony_train_list.txt").absolute(),
            pack_fn=snoy_pack,
            transform=transfrom,
        )
        self.valid_dataset = DataSetSID(
            config=data_cfg,
            file=Path(data_cfg.data_root).joinpath("Sony_val_list.txt").absolute(),
            pack_fn=snoy_pack,
            transform=transfrom,
        )
        self.pred_dataset = DataSetSID(
            config=data_cfg,
            file=Path(data_cfg.data_root).joinpath("Sony_test_list.txt").absolute(),
            pack_fn=snoy_pack,
            transform=transfrom,
        )
        x, y = self.train_dataset.__getitem__(0)
        print(x.shape)
        print(y.shape)
