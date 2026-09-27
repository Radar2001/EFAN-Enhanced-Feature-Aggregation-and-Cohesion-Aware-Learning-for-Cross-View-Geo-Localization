"""Training script for EFAN on the University-1652 dataset."""

import os
import shutil
import sys
import time
from dataclasses import dataclass

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup)

from dataset import U1652DatasetEval, U1652DatasetTrain, get_transforms
from evaluate import evaluate
from loss import CAAHL
from model import EFAN
from trainer import train
from utils import Logger, setup_system


@dataclass
class Configuration:
    net: str = 'EFAN'
    # Model
    model: str = 'swinv2_base_window12to24_192to384'

    psm: bool = True
    img_size: int = 384
    image_size_sat = (img_size, img_size)
    img_size_ground = (img_size, img_size)

    # Training
    mixed_precision: bool = True
    custom_sampling: bool = True  # use custom sampling instead of random
    seed = 1
    epochs: int = 5
    batch_size: int = 2  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0)  # GPU ids for training

    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 1  # eval every n epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all or int

    # Optimizer
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.0001
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.5
    lr_end: float = 0.0001  # only for "polynomial"

    # Dataset
    dataset: str = 'U1652-D2S'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"

    # Augment Images
    prob_flip: float = 0.5  # flip sat/drone image simultaneously

    # Savepath for model checkpoints
    model_path: str = "University"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 16

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = Configuration()

if config.dataset == 'U1652-D2S':
    config.query_folder_train = config.data_folder + '/train/satellite'
    config.gallery_folder_train = config.data_folder + '/train/drone'
    config.query_folder_test = config.data_folder + '/test/query_drone'
    config.gallery_folder_test = config.data_folder + '/test/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = config.data_folder + '/train/satellite'
    config.gallery_folder_train = config.data_folder + '/train/drone'
    config.query_folder_test = config.data_folder + '/test/query_satellite'
    config.gallery_folder_test = config.data_folder + '/test/gallery_drone'

if __name__ == '__main__':
    model_path = "{}/{}/{}_{}_{}".format(
        config.model_path,
        config.model,
        config.net,
        config.dataset,
        time.strftime("%m-%d-%H-%M-%S"),
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#
    print(f"Using seed: {config.seed}")
    print("\nModel: {}".format(config.model))

    model = EFAN(config.model, img_size=config.img_size)

    data_config = model.get_config()
    print(data_config)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size_sat = config.image_size_sat
    img_size_ground = config.img_size_ground

    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    img_size = (config.img_size, config.img_size)
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    train_dataset = U1652DatasetTrain(
        query_folder=config.query_folder_train,
        gallery_folder=config.gallery_folder_train,
        transforms_query=train_sat_transforms,
        transforms_gallery=train_drone_transforms,
        prob_flip=config.prob_flip,
        shuffle_batch_size=config.batch_size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=not config.custom_sampling,
        pin_memory=True,
    )

    query_dataset_test = U1652DatasetEval(
        data_folder=config.query_folder_test,
        mode="query",
        transforms=val_transforms,
    )

    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    gallery_dataset_test = U1652DatasetEval(
        data_folder=config.gallery_folder_test,
        mode="gallery",
        transforms=val_transforms,
        sample_ids=query_dataset_test.get_sample_ids(),
        gallery_n=config.eval_gallery_n,
    )

    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    loss_function = CAAHL(
        loss_function=loss_fn,
        base_temperature=0.07,
        circle_margin=0.3,
        circle_gamma=128,
        initial_circle_weight=0.6,
        device=config.device,
    )

    if config.mixed_precision:
        scaler = torch.amp.GradScaler('cuda', init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # Optimizer                                                                   #
    # -----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        scale_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'global_scale_logit' in name:
                scale_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': config.lr},
            {'params': scale_params, 'lr': config.lr * 50, 'weight_decay': 0.0},
        ])

    # -----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    # -----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=config.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps,
        )
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps,
        )
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    # -----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))
        r1_test = evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True,
        )

    # -----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    # -----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    best_score = 0

    for epoch in range(1, config.epochs + 1):
        if hasattr(loss_function, 'set_epoch'):
            loss_function.set_epoch(epoch - 1)

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        train_loss = train(
            config,
            model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(
            epoch, train_loss, optimizer.param_groups[0]['lr']))

        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            r1_test = evaluate(
                config=config,
                model=model,
                query_loader=query_dataloader_test,
                gallery_loader=gallery_dataloader_test,
                ranks=[1, 5, 10],
                step_size=1000,
                cleanup=True,
            )

            if r1_test > best_score:
                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
