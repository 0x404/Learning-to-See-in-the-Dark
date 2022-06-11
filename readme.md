# Learning to See in the Dark
* [中文版本](./doc/readme.md)

* this is an implementation of [SID](https://arxiv.org/abs/1805.01934), based on my deeplearning framework [Rainy](https://github.com/0x404/Rainy)

## Train

complete config file according to config template, then launch.

the default setting will save top 10 models based on PSNR, and every time running validating, it will save a comparison picture between model output and ground truth.

all parameters mentioned by the article can be configured, see [config template](./configs/sid.py).

``` shell
python3 launch.py --config configs/sid.py
```

## test

complete config file according to [config template](./configs/sid.py), set `use_path` to False, `do_train` to False, `do_predict` to True.

```shell
python3 launch.py --config configs/sid.py
```

or you can just set by command line.

```shell
python3 launch.py --config configs/sid.py --do_predict --init_checkpoint ./checkpoint/yourcheckpoint
```


