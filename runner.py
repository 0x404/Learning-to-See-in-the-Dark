"""runner for training loop"""
# pylint: disable=logging-fstring-interpolation
from ctypes import util
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import task
import utils
import numpy as np
from utils import get_logger, move_to_device, Saver
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

logger = get_logger(__name__)


class Runner:
    """Base Runner"""

    def __init__(self, config):
        self.config = config
        self.task = getattr(task, config.task.name, None)
        if self.task is None:
            raise ValueError(f"task {config.task.name} is not supported!")
        self.task = self.task(config)
        self.model = self.task.model
        self.model_saver = Saver(self.model, config)
        self.device = torch.device(config.setup.device)
        self.completed_step = 0
        logger.info(f"training on {self.device}")

        if config.train.init_checkpoint is not None:
            self.model_saver.resume_from_file(config.train.init_checkpoint)
        self.writer = None
        if config.setup.tensorboard:
            self.writer = SummaryWriter()

    def train(self):
        """Do training loop.

        The Runner fetch model, optimizer and loos_function from task config,
        and do the training loop, evaluation, logging and summary write.
        """
        config = self.config.train
        setup = self.config.setup

        model = self.model
        optimizer = self.task.optimizer
        optimizer.zero_grad()
        loss_function = self.task.loss_function
        lr_scheduler = getattr(self.task, "lr_scheduler", None)
        train_loader = DataLoader(
            dataset=self.task.train_dataset,
            batch_size=config.batch_size,
            num_workers=setup.get("data_worker_num", 1),
            collate_fn=getattr(self.task, "collate_fn", None),
            shuffle=True,
        )

        total_step = config.max_step
        if total_step is None:
            total_step = config.epochs * len(train_loader)
        total_step = min(total_step, config.epochs * len(train_loader))

        logger.info("********** Running training **********")
        logger.info(f"  Num Examples = {len(train_loader)}")
        logger.info(f"  Num Epochs = {config.epochs}")
        logger.info(f"  Global Total Step = {total_step}")
        logger.info(f"  Train Batch Size = {config.batch_size}")
        logger.info(f"  Accumulate Gradient Step = {config.accumulate_step}")
        logger.info(f"  Model Structure = {model}")

        for epoch in range(config.epochs):
            for step, batch_data in enumerate(train_loader):
                inputs, labels = batch_data
                inputs = move_to_device(inputs, self.device)
                labels = move_to_device(labels, self.device)

                if epoch == 0 and step == 0:
                    if isinstance(inputs, (dict, list)):
                        logger.info(f"Input: {inputs}")
                    else:
                        logger.info(f"Input Shape: {inputs.shape}")
                        logger.info(f"Input Dtype: {inputs.dtype}")
                    logger.info(f"Labels Shape: {labels.shape}")
                    logger.info(f"Labels Dtype: {labels.dtype}")

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()

                if step % config.accumulate_step == 0 or step == len(train_loader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    self.completed_step += 1
                    if self.writer is not None:
                        self.writer.add_scalar("loss", loss.item(), self.completed_step)

                if (
                    self.completed_step % setup.log_every_n_step == 0
                    or self.completed_step == total_step
                ):
                    if self.completed_step == 0:
                        continue
                    progress_tag = 100 * self.completed_step / total_step
                    logger.info(
                        f"[{progress_tag:.2f}%]\t epoch:{epoch}\t step:{self.completed_step}\t loss:{loss.item()}"
                    )

                if self.completed_step % setup.save_ckpt_n_step == 0:
                    if self.completed_step == 0:
                        continue
                    accuracy = self.eval(is_training=True)
                    self.model_saver.save_model(accuracy)

                if self.completed_step >= total_step:
                    logger.info(
                        f"reach max training step {total_step}, breaking from training loop."
                    )
                    if self.writer is not None:
                        self.writer.flush()
                        self.writer.close()
                    break

            if self.completed_step >= total_step:
                break
            if lr_scheduler is not None:
                lr_scheduler.step()

    def eval(self, is_training=False):
        """Evaluation

        Args:
            is_training (bool, optional): whther is training. Defaults to False.

        Returns:
            float: the accuracy of this evaluation.
        """
        config = self.config.predict
        eval_loader = DataLoader(
            dataset=self.task.valid_dataset,
            batch_size=config.batch_size,
            num_workers=self.config.setup.get("data_worker_num", 1),
            collate_fn=getattr(self.task, "collate_fn", None),
        )
        eval_type = "Evalution" if is_training else "Prediction"
        logger.info(f"********** Running {eval_type} **********")
        logger.info(f"  Num Examples = {len(eval_loader)}")
        logger.info(f"  Batch Size = {eval_loader.batch_size}")

        psnr_meter = utils.AverageValueMeter("psnr")
        ssim_meter = utils.AverageValueMeter("ssim")

        image_saved = False
        with torch.no_grad():
            for idx, batch_data in enumerate(tqdm(eval_loader, desc=eval_type)):
                inputs, labels = batch_data
                inputs = move_to_device(inputs, self.device)
                labels = move_to_device(labels, self.device)
                outputs = self.model(inputs)

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
                    if idx == 0 and not image_saved:
                        compare_img = np.concatenate(
                            (output[:, :, :], label[:, :, :]), axis=1
                        )
                        savepath = Path(self.config.predict.output_root).joinpath(
                            f"compare_step{self.completed_step}.jpg"
                        )
                        utils.toimage(
                            compare_img, high=255, low=0, cmax=255, cmin=0
                        ).save(savepath)
                        image_saved = True

        logger.info(
            f"{eval_type} finished! mean psnr = {psnr_meter.mean}, mean ssim = {ssim_meter.mean}"
        )
        if self.writer is not None:
            self.writer.add_scalar(
                psnr_meter.name, psnr_meter.mean, self.completed_step
            )
            self.writer.add_scalar(
                ssim_meter.name, ssim_meter.mean, self.completed_step
            )
        return psnr_meter.mean

    def predict(self):
        """Do prediction.

        Load best model from checkpoint and do evaluation,
        then run the test dataset and write predictions to `predict_path`.
        """
        config = self.config.predict
        self.model_saver.load_best_model()
        self.eval()

        pred_loader = DataLoader(
            dataset=self.task.pred_dataset,
            batch_size=1,
            collate_fn=getattr(self.task, "collate_fn", None),
        )
        with torch.no_grad():
            for idx, data in enumerate(tqdm(pred_loader, desc="test")):
                inputs, _ = data
                inputs = move_to_device(inputs, self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()

                outputs = torch.clamp(outputs, 0, 1).cpu()
                outputs = outputs.numpy().transpose(1, 2, 0) * 255.0
                filename = Path(f"{config.output_root}").joinpath(f"result_{idx}.jpg")
                utils.toimage(outputs, high=255, low=0, cmax=255, cmin=0).save(
                    str(filename)
                )
