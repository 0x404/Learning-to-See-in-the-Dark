import rawpy
import torch
import numpy as np
import PIL.Image
from pathlib import Path
from torch.utils.data import Dataset


def filp_transform(image, config):
    """Fliping an image by probility.

    Args:
        image (numpy.ndarray): with shape [C, H, W].
        config (Config): transform config.

    Returns:
        numpy.ndarray: with shape [C, H, W].
    """
    # filp horizontally
    if np.random.random() < config.filp_prob:
        image = np.flip(image, axis=2)
    # filp vertically
    if np.random.random() < config.filp_prob:
        image = np.flip(image, axis=1)
    return image


def rotation_transform(image, config):
    """Rotation an image by probility.

    Args:
        image (numpy.ndarray): with shape [C, H, W].
        config (Config): transform config.

    Returns:
        numpy.ndarray: with shape [C, W, H] by probility.
    """
    if np.random.random() < config.rotation_prob:
        image = np.transpose(image, (0, 2, 1))
    return image


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

    def __init__(self, config, file, pack_fn, transform=None):
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
        self.pack_fn = pack_fn
        self.transform = transform
        self.data = []
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

        input_exposure = float(input_path.stem.split("_")[-1][:-1])
        label_exposure = float(label_path.stem.split("_")[-1][:-1])

        if self.config.use_constant_amplification:
            amplification_ratio = self.config.amplification_ratio
        else:
            amplification_ratio = min(
                label_exposure / input_exposure, self.config.amplification_ratio
            )

        self.data.append(
            {
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
        item = self.data[index]
        # read input image
        print(item["input_path"])
        image = rawpy.imread(str(item["input_path"]))
        image = self.pack_fn(image) * item["ratio"]
        image = image.transpose(2, 0, 1)

        # read label image
        if self.config.use_png_label:
            label = np.array(PIL.Image.open(str(item["label_path"])), dtype=np.float32)
            label = label / 255.0
            label = label.transpose(2, 0, 1)
        else:
            process_cfg = self.config.raw_process
            label = rawpy.imread(str(item["label_path"]))
            label = label.postprocess(
                use_camera_wb=process_cfg.use_camera_wb,
                no_auto_bright=process_cfg.no_auto_bright,
                half_size=process_cfg.half_size,
                output_bps=process_cfg.output_bps,
            )
            label = np.float32(label / 65535.0)
            label = label.transpose(2, 0, 1)

        # apply patch transform
        if self.config.transform.use_patch:
            patch_size = self.config.transform.patch_size
            H, W = image.shape[1], image.shape[2]
            rand_x = np.random.randint(0, H - patch_size)
            rand_y = np.random.randint(0, W - patch_size)
            image = image[:, rand_x : rand_x + patch_size, rand_y : rand_y + patch_size]
            label = label[
                :,
                rand_x * 2 : rand_x * 2 + patch_size * 2,
                rand_y * 2 : rand_y * 2 + patch_size * 2,
            ]

        # apply other transform
        if self.transform is not None:
            for t in self.transform:
                image = t(image, self.config.transform)

        # move to tensor
        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).float()
        return image, label
   