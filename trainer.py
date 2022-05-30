import torch
from torch import optim
from model import UNet
from dataset import DataSetSID
from dataset.sid import snoy_pack, filp_transform, rotation_transform
from pathlib import Path
from torch.utils.data import DataLoader
import rawpy
import numpy as np
from utils import get_logger, Saver
from tqdm import tqdm
import utils
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__)


def collate_fn(batch):
    return batch


label_image = [None for _ in range(25000)]
input_with_ratio = {
    "100": [None for _ in range(25000)],
    "250": [None for _ in range(25000)],
    "300": [None for _ in range(25000)],
}

class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.setup.device
        self.model = UNet(config.model.channel_in, config.model.channel_out)
        self.model.to(self.device)
        self.saver = Saver(self.model, config)
        if config.train.init_checkpoint is not None:
            self.saver.resume_from_file(config.train.init_checkpoint)
        self.writer = None
        if config.setup.tensorboard:
            self.writer = SummaryWriter()

    def load_data(self, item, config, transfroms=None, fixed=False):
        image_id = item["id"]
        ratio = item["ratio"]
        label_path = item["label_path"]
        input_path = item["input_path"]

        # load and cache input image
        if input_with_ratio[str(ratio)[:3]][image_id] is None:
            raw = rawpy.imread(str(input_path))
            raw = snoy_pack(raw)
            raw = np.expand_dims(raw, axis=0) * ratio
            input_with_ratio[str(ratio)[:3]][image_id] = raw

        # load and cache label image
        if label_image[image_id] is None:
            image = rawpy.imread(str(label_path))
            image = image.postprocess(
                use_camera_wb=config.raw_process.use_camera_wb,
                no_auto_bright=config.raw_process.no_auto_bright,
                half_size=config.raw_process.half_size,
                output_bps=config.raw_process.output_bps,
            )
            image = np.float32(image / 65535.0)
            image = np.expand_dims(image, axis=0)
            label_image[image_id] = image

        # apply patch
        if config.transform.use_patch:
            # the reason why we make label patch [3, patch_size * 2, patch_size * 2]
            # is that input of model is [4, patch_size, patch_size]
            # and the output of model is [12, patch_size, patch_size]
            # and we use L1loss as criterion function,
            # so we need to label element sum equals output element sum
            patch_size = config.transform.patch_size
            H = input_with_ratio[str(ratio)[:3]][image_id].shape[1]
            W = input_with_ratio[str(ratio)[:3]][image_id].shape[2]
            rand_x = np.random.randint(0, H - patch_size)
            rand_y = np.random.randint(0, W - patch_size)
            if fixed:
                rand_x = 500
                rand_y = 500
            input = input_with_ratio[str(ratio)[:3]][image_id][
                :, rand_x : rand_x + patch_size, rand_y : rand_y + patch_size, :
            ]
            label = label_image[image_id][
                :,
                rand_x * 2 : rand_x * 2 + patch_size * 2,
                rand_y * 2 : rand_y * 2 + patch_size * 2,
                :,
            ]

        # apply transfrom
        if transfroms is not None:
            for trans in transfroms:
                input, label = trans(input, label, config.transform)

        input = torch.from_numpy(input.copy())
        label = torch.from_numpy(label.copy())
        input = input.permute(0, 3, 1, 2).to(self.device)
        label = label.permute(0, 3, 1, 2).to(self.device)
        return input, label

    def train(self):
        self.eval(True, 0)
        traincfg = self.config.train
        datacfg = self.config.data
        setup = self.config.setup
        loss_function = torch.nn.L1Loss()
        loss_function.to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=traincfg.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=traincfg.lr_step,
            gamma=traincfg.lr_gamma,
        )

        dataset = DataSetSID(
            config=datacfg,
            file=Path(datacfg.data_root).joinpath("Sony_train_list.txt").absolute(),
        )
        dataloader = DataLoader(dataset, collate_fn=collate_fn)

        transforms = []
        if datacfg.transform.use_flip:
            transforms.append(filp_transform)
        if datacfg.transform.use_rotation:
            transforms.append(rotation_transform)

        total_step = traincfg.max_step
        if total_step is None:
            total_step = traincfg.epochs * len(dataloader)
        total_step = min(total_step, traincfg.epochs * len(dataloader))
        completed_step = 0
        for epoch in range(traincfg.epochs):
            for _, data in enumerate(dataloader):
                # print(_)
                input, label = self.load_data(data[0], datacfg, transforms)
                output = self.model(input)

                loss = loss_function(output, label)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                completed_step += 1

                if (
                    completed_step % setup.log_every_n_step == 0
                    or completed_step == total_step
                ):
                    progress_tag = 100 * completed_step / total_step
                    logger.info(
                        f"[{progress_tag:.2f}%]\t epoch:{epoch}\t step:{completed_step}\t loss:{loss.item()}"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("loss", loss.item(), completed_step)

                if completed_step % setup.save_ckpt_n_step == 0:
                    psnr = self.eval(is_training=True, global_step=completed_step)
                    self.saver.save_model(psnr)
            lr_scheduler.step()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def eval(self, is_training=False, global_step=0):
        datacfg = self.config.data
        dataset = DataSetSID(
            config=datacfg,
            file=Path(datacfg.data_root).joinpath("Sony_val_list.txt").absolute(),
        )
        dataloader = DataLoader(dataset, collate_fn=collate_fn)

        eval_type = "validating" if is_training else "predicting"
        psnr_meter = utils.AverageValueMeter("psnr")
        ssim_meter = utils.AverageValueMeter("ssim")

        with torch.no_grad():
            image_saved = False
            for _, data in enumerate(tqdm(dataloader, desc=eval_type)):

                input, labels = self.load_data(
                    data[0], datacfg, transfroms=None, fixed=True
                )
                outputs = self.model(input)

                outputs = torch.clamp(outputs, 0, 1).cpu()
                labels = labels.cpu()

                for output, label in zip(outputs, labels):
                    output = output.numpy().transpose(1, 2, 0) * 255.0
                    label = label.numpy().transpose(1, 2, 0) * 255.0
                    psnr = compare_psnr(output, label, data_range=255)
                    ssim = compare_ssim(
                        output,
                        label,
                        data_range=255,
                        gaussian_weights=True,
                        use_sample_covariance=False,
                        multichannel=True,
                    )
                    psnr_meter.add(psnr, 1)
                    ssim_meter.add(ssim, 1)
                    if is_training and not image_saved:
                        compare_img = np.concatenate(
                            (output[:, :, :], label[:, :, :]), axis=1
                        )
                        savepath = Path(self.config.predict.output_root).joinpath(
                            f"compare_step{global_step}.jpg"
                        )
                        utils.toimage(
                            compare_img, high=255, low=0, cmax=255, cmin=0
                        ).save(savepath)
                        image_saved = True
            logger.info(
                f"{eval_type} finished! mean psnr = {psnr_meter.mean}, mean ssim = {ssim_meter.mean}"
            )
            if is_training and self.writer is not None:
                self.writer.add_scalar(psnr_meter.name, psnr_meter.mean, global_step)
                self.writer.add_scalar(ssim_meter.name, ssim_meter.mean, global_step)
            return psnr_meter.mean
