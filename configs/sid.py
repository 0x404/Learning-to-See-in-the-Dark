import os
import torch

setup = dict(
    do_train=False,
    do_predict=False,
    tensorboard=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_checkpoints=3,
    checkpoint_path=os.path.join("checkpoints", "test"),
    log_every_n_step=1,
    save_ckpt_n_step=5,
)
task = dict(name="SeeInDark")
data = dict(
    camera="sony",
    data_root="sony/",
    # if use constant amplification, then `ratio =  amplification ratio`
    # if not use constant amplification, then `ratio = min(actual amplification, amplification ratio)`
    use_constant_amplification=False,
    amplification_ratio=300,
    # whther to transform label iamge to png
    # if use png label, then provide png label root
    use_png_label=False,
    png_root="",
    # use transform to enhance data
    transform=dict(
        use_patch=True,
        patch_size=512,
        use_flip=True,
        filp_prob=0.5,
        use_rotation=True,
        rotation_prob=0.5,
    ),
    # config of rawpy when processing raw iamge
    raw_process=dict(
        use_camera_wb=False,
        no_auto_bright=False,
        half_size=False,
        output_bps=16,
    ),
    # load limit
    limit=None,
)
train = dict(
    lr=1e-4,  # become 1e-5 when epoch = 2000
    lr_step=2000,
    lr_gamma=0.1,
    batch_size=32,
    epochs=4000,
    accumulate_step=1,
    init_checkpoint=None,
    max_step=None,
)
predict = dict(batch_size=32, output_root="predictions")

model = dict(
    channel_in=4,
    channel_out=12,
)
