from cProfile import label
import rawpy
import torch
import numpy as np
import PIL.Image
from pathlib import Path
from torch.utils.data import Dataset


def filp_transform(image1, image2, config):
    """Fliping an image by probility.

    Args:
        image (numpy.ndarray): with shape [B, H, W, C].
        config (Config): transform config.

    Returns:
        numpy.ndarray: with shape [B, H, W, C].
    """
    # filp horizontally
    if np.random.random() < config.filp_prob:
        image1 = np.flip(image1, axis=2)
        image2 = np.flip(image2, axis=2)
    # filp vertically
    if np.random.random() < config.filp_prob:
        image1 = np.flip(image1, axis=1)
        image2 = np.flip(image2, axis=1)
    return image1, image2


def rotation_transform(image1, image2, config):
    """Rotation an image by probility.

    Args:
        image (numpy.ndarray): with shape [B, H, W, C].
        config (Config): transform config.

    Returns:
        numpy.ndarray: with shape [B, W, H, C] by probility.
    """
    if np.random.random() < config.rotation_prob:
        image1 = np.transpose(image1, (0, 2, 1, 3))
        image2 = np.transpose(image2, (0, 2, 1, 3))
    return image1, image2


def snoy_pack(raw):
    """Pack bayer image into 4 channels and correspondingly reduce
    the spatial resolution by a factor of two in each dimension.
    see `arxiv.org/abs/1805.01934` 's explanation of sony pipeline.

    Args:
        raw (rawpy.Rawpy): raw image readed by rawpy,
                           with shape [H, W, 3].

    Returns:
        numpy.ndarray: with shape [H/2, W/2, 4]
    """
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)

    # subtract the black level
    im = np.maximum(im - 512, 0) / (16383 - 512)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    return out


class DataSetSID(Dataset):
    """SID Dataset"""

    def __init__(self, config, file):
        """SID Dataset.

        Args:
            config (Config): data config.
            file (str or Path): data file, e.g. `Sony_train_list.txt`
            pack_fn (callable): callable object to pack raw bayer data.
            transform ([callable], optional): transfrom applied to input data. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.root = Path(config.data_root)
        self.png_root = Path(config.png_root) if config.use_png_label else None
        self.data = []
        self.fname2id = {}
        self.iamge_cache = [None] * 6000
        # input-demo: http://data-rainy.oss-cn-beijing.aliyuncs.com/data/SID-demo/00001_00_0.1s.ARW
        # label-demo: http://data-rainy.oss-cn-beijing.aliyuncs.com/data/SID-demo/00001_00_10s.ARW

        with open(file, mode="r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split()
                img_input, img_label = line[:2]
                self._upadte_data(img_input, img_label)
                if config.limit is not None and idx == config.limit:
                    break

    def _upadte_data(self, raw_img_input, raw_img_label):
        """Update dataset by one item.

        Args:
            raw_img_input (str): input iamge described in data file.
            raw_img_label (str): label iamge described in data file.
        """
        input_path = self.root.joinpath(Path(raw_img_input[7:])).absolute()
        if self.png_root is not None:
            label_path = self.png_root.joinpath(Path(raw_img_label[7:])).absolute()
        else:
            label_path = self.root.joinpath(Path(raw_img_label[7:])).absolute()

        if input_path not in self.fname2id:
            self.fname2id[str(input_path)] = len(self.fname2id) + 1
        if label_path not in self.fname2id:
            self.fname2id[str(label_path)] = len(self.fname2id) + 1

        input_exposure = float(input_path.stem.split("_")[-1][:-1])
        label_exposure = float(label_path.stem.split("_")[-1][:-1])
        id = int(label_path.stem[:5])

        if self.config.use_constant_amplification:
            amplification_ratio = self.config.amplification_ratio
        else:
            amplification_ratio = min(
                label_exposure / input_exposure, self.config.amplification_ratio
            )

        self.data.append(
            {
                "id": id,
                "input_path": input_path,
                "label_path": label_path,
                "input_exposure": input_exposure,
                "label_exposure": label_exposure,
                "ratio": amplification_ratio,
            }
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]