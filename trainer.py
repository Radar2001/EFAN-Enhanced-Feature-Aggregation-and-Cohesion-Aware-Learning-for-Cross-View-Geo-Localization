"""Training and inference loops."""

import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from utils import AverageMeter


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # zero gradients for the first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader), ncols=150, position=0, leave=True)
    else:
        bar = dataloader

    for query, reference, ids in bar:
        if scaler:
            with autocast():
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp())
                losses.update(loss.item())

            scaler.scale(loss).backward()

            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if train_config.scheduler in ("polynomial", "cosine", "constant"):
                scheduler.step()

        else:
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            loss.backward()

            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            optimizer.step()
            optimizer.zero_grad()

            if train_config.scheduler in ("polynomial", "cosine", "constant"):
                scheduler.step()

        if train_config.verbose:
            if len(optimizer.param_groups) > 1:
                monitor = {
                    "loss": "{:.4f}".format(loss.item()),
                    "loss_avg": "{:.4f}".format(losses.avg),
                    "lr1": "{:.6e}".format(optimizer.param_groups[0]['lr']),
                    "lr2": "{:.6e}".format(optimizer.param_groups[1]['lr']),
                }
            else:
                monitor = {
                    "loss": "{:.4f}".format(loss.item()),
                    "loss_avg": "{:.4f}".format(losses.avg),
                    "lr": "{:.6e}".format(optimizer.param_groups[0]['lr']),
                }
            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader, is_autocast=True, input_id=1):
    model.eval()

    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader), ncols=100, position=0, leave=True)
    else:
        bar = dataloader

    img_features_list = []
    ids_list = []

    with torch.no_grad():
        for img, ids in bar:
            ids_list.append(ids)

            if is_autocast:
                with autocast():
                    img = img.to(train_config.device)
                    img_feature = model(img, input_id=input_id)
            else:
                img = img.to(train_config.device)
                img_feature = model(img, input_id=input_id)

            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features_list.append(img_feature.to(torch.float32))

        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    return img_features, ids_list
