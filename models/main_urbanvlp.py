import math
import os
import time
import json
import argparse
import numpy as np
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
import open_clip_mine as open_clip
from utils import (
    MultigranuDataset_full_preload,
    set_random_seed,
    unwrap_model,
    get_clip_metrics,
    get_GPU_usage,
)
from tqdm import tqdm
from loguru import logger
import wandb
import datetime
from models import MultiGranularity_GeoCLIP, MultiGranularity_GeoCLIP_GateFusion

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        choices=["Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/urbansv",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=3e-4, 
        help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="weight decay")
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=2, 
        help="batch size"
    )
    parser.add_argument(
        "--epoch_num", 
        type=int, 
        default=10, 
        help="epoch number"
        )
    parser.add_argument(
        "--log_every_n_steps",
        type=int, 
        default=100, 
        help="log every n steps"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default = None,
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=132, 
        help="random seed"
    )
    parser.add_argument(
        "--logging_dir", 
        type=str, 
        default="logs", 
        help="logging directory"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints", 
        help="checkpoint path"
    )
    # data hyper-parameters
    parser.add_argument(
        "--train_dataset_ratio",
        type=float,
        default=0.8,
        help="ratio of training dataset",
    )
    parser.add_argument(
        "--val_dataset_ratio",
        type=float,
        default=0.1,
        help="ratio of validation dataset",
    )
    parser.add_argument(
        "--test_dataset_ratio", 
        type=float, 
        default=0.1, 
        help="ratio of test dataset"
    )
    parser.add_argument(
        "--close_wandb",
        action='store_true',
    )
    parser.add_argument(
        "--wandb_id", 
        type=str, 
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="urbanvlp",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
    )
    
    parser.add_argument(
        "--gatefusion",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    return args


def create_datasets(args, transform, tokenizer):
    """To create train, val, test datasets."""
    # assert args.dataset in ['Beijing','Shanghai','Guangzhou','Shenzhen']
    data = json.load(open(f"data/urbansat/satellite_texts/{args.dataset}.json", "r"))

    # split dataset into train, val, test
    np.random.shuffle(data)
    train_data = data[: int(len(data) * args.train_dataset_ratio)]
    val_data = data[
        int(len(data) * args.train_dataset_ratio) : int(
            len(data) * (args.train_dataset_ratio + args.val_dataset_ratio)
        )
    ]
    test_data = data[
        int(len(data) * (args.train_dataset_ratio + args.val_dataset_ratio)) :
    ]

    # create datasets
    train_dataset = MultigranuDataset_full_preload(
                        train_data, 
                        transform, 
                        tokenizer,
                        return_coordinate=True,
                        data_path=args.data_path,
                        city=args.dataset.split('_')[0],
                    )
    val_dataset = MultigranuDataset_full_preload(
                        val_data, 
                        transform, 
                        tokenizer,
                        return_coordinate=True,
                        data_path=args.data_path,
                        city=args.dataset.split('_')[0],
                    )
    test_dataset = MultigranuDataset_full_preload(
                        test_data, 
                        transform, 
                        tokenizer,
                        return_coordinate=True,
                        data_path=args.data_path,
                        city=args.dataset.split('_')[0],
                    )

    return train_dataset, val_dataset, test_dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
                        model, 
                        criterion, 
                        data, 
                        epoch, 
                        optimizer, 
                        args, 
                        logger,
                    ):
    """To train one epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    dataloader = data["train_loader"]
    num_batches_per_epoch = len(dataloader)
    sample_digits = math.ceil(math.log(len(dataloader) * args.batch_size + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()  # data loading time
    end = time.time()
    for batch_count, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + batch_count
        (
            images,
            texts,
            streetview_images,
            streetview_texts,
            streetview_coordinates,
        ) = batch  
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        streetview_images = streetview_images.to(device=device, non_blocking=True)
        streetview_texts = streetview_texts.to(device=device, non_blocking=True)
        streetview_coordinates = streetview_coordinates.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        if texts.ndim == 3:
            texts = texts.squeeze(1)
            streetview_texts = streetview_texts.squeeze(1)
        satellite_model_out, streetview_model_out = model(
                        images,
                        texts,
                        streetview_images,
                        streetview_texts,
                        streetview_coordinates,
        )
        
        logit_scale = satellite_model_out["logit_scale"]
        losses_satellite = criterion(
                image_features = satellite_model_out['image_features'],
                text_features = satellite_model_out['text_features'],
                image_all_tokens = None,
                text_all_tokens = None,
                logit_scale = logit_scale, 
                output_dict=True,
        )
        losses_streetview = criterion(
                image_features = streetview_model_out['image_features'],
                text_features = streetview_model_out['text_features'],
                image_all_tokens = streetview_model_out['image_all_tokens'],
                text_all_tokens = streetview_model_out['text_all_tokens'],
                logit_scale = logit_scale, 
                output_dict=True,
        )
        total_loss = args.alpha*sum(losses_satellite.values()) + args.beta*sum(losses_streetview.values())
        losses = {}
        losses["loss"] = total_loss
        losses["contrastive_loss_satellite"] = losses_satellite['contrastive_loss']
        losses["contrastive_loss_streetview"] = losses_streetview['contrastive_loss']

        total_loss.backward()

        optimizer.step()

        with torch.no_grad():
            unwrap_model(model.model_satellite).logit_scale.clamp_(0, math.log(100))
            unwrap_model(model.model_streetview).logit_scale.clamp_(0, math.log(100))
            

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count += 1
        if step % args.log_every_n_steps == 0:
            batch_size = len(images)
            num_samples = step * batch_size
            samples_per_epoch = (
                num_batches_per_epoch * batch_size
            )
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            dict_ = {}
            dict_['epoch'] = epoch
            for loss_name, loss_m in losses_m.items():
                dict_[f"train/{loss_name.capitalize()}"] = loss_m.val
            wandb.log(dict_)

            samples_per_second = batch_size / batch_time_m.val
            logger.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )
            get_GPU_usage()
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "scale": logit_scale_scalar,
            }
            log_data.update({name: val.val for name, val in losses_m.items()})
            for name, val in log_data.items():
                name = "train/" + name
                logger.info({name: val, "step": step})

            batch_time_m.reset()
            data_time_m.reset()


def evaluate(
                model,
                data, 
                epoch, 
                args, 
                logger, 
            ):
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = data["val_loader"]
    num_samples = 0
    samples_per_val = len(dataloader) * args.batch_size  # sample size per epoch

    cumulative_loss = 0.0
    cumulative_gen_loss = 0.0
    all_satellite_image_features, all_satellite_text_features = [], []
    all_streetview_image_features, all_streetview_text_features = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (
                images,
                texts,
                streetview_images,
                streetview_texts,
                streetview_coordinates,
            ) = batch
                    
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            streetview_images = streetview_images.to(device=device, non_blocking=True)
            streetview_texts = streetview_texts.to(device=device, non_blocking=True)
            streetview_coordinates = streetview_coordinates.to(device=device, non_blocking=True)
            
            if texts.ndim == 3:
                texts = texts.squeeze(1)
                streetview_texts = streetview_texts.squeeze(1)
            
            satellite_model_out, streetview_model_out = model(
                            images,
                            texts,
                            streetview_images,
                            streetview_texts,
                            streetview_coordinates,
            )
            
            satellite_image_features = satellite_model_out["image_features"]
            satellite_text_features = satellite_model_out["text_features"]
            satellite_logit_scale = satellite_model_out["logit_scale"]


            streetview_image_features = streetview_model_out["image_features"]
            streetview_text_features = streetview_model_out["text_features"]
            
            all_satellite_image_features.append(satellite_image_features.cpu())
            all_satellite_text_features.append(satellite_text_features.cpu())
            
            all_streetview_image_features.append(streetview_image_features.cpu())
            all_streetview_text_features.append(streetview_text_features.cpu())
            
            logit_scale = satellite_logit_scale.mean()
            logits_per_satellite_image = logit_scale * satellite_image_features @ satellite_text_features.t()
            logits_per_satellite_text = logits_per_satellite_image.t()
            
            logits_per_streetview_image = logit_scale * streetview_image_features @ streetview_text_features.t()
            logits_per_streetview_text = logits_per_streetview_image.t()

            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=device).long()
            total_loss_satellite = (  
                F.cross_entropy(logits_per_satellite_image, labels)
                + F.cross_entropy(logits_per_satellite_text, labels)
            ) / 2
            
            total_loss_streetview = (  
                F.cross_entropy(logits_per_streetview_image, labels)
                + F.cross_entropy(logits_per_streetview_text, labels)
            ) / 2

            cumulative_loss += (total_loss_satellite + total_loss_streetview) * batch_size
            num_samples += batch_size

            if i % 100 == 0:
                logger.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                )

        satellite_val_metrics = get_clip_metrics(
            image_features=torch.cat(all_satellite_image_features),
            text_features=torch.cat(all_satellite_text_features),
            logit_scale=logit_scale.cpu(),
            prefix='satellite'
        )
        
        streetview_val_metrics = get_clip_metrics(
            image_features=torch.cat(all_streetview_image_features),
            text_features=torch.cat(all_streetview_text_features),
            logit_scale=logit_scale.cpu(),
            prefix='streetview'
        )
        loss = cumulative_loss / num_samples
        metrics.update(
            {
                **satellite_val_metrics,
                **streetview_val_metrics,
                "clip_val_loss": loss.item(),
                "epoch": epoch,
                "num_samples": num_samples,
            }
        )

    logger.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    dict_ = {}
    dict_['epoch']=epoch
    for name, val in metrics.items():
        logger.info({f"val/{name}": val, "epoch": epoch})
        dict_[f"val/{name}"] = val
    wandb.log(dict_)
    return metrics


def inference(model, data, args, logger):
    """test on test dataset."""
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = data["test_loader"]
    num_samples = 0
    samples_per_val = len(dataloader) * args.batch_size

    cumulative_loss = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            if texts.ndim == 3:
                texts = texts.squeeze(1)
            model_out = model(images, texts)
            image_features = model_out["image_features"]
            text_features = model_out["text_features"]
            logit_scale = model_out["logit_scale"]

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=device).long()
            total_loss = (  # contrastive loss
                F.cross_entropy(logits_per_image, labels)
                + F.cross_entropy(logits_per_text, labels)
            ) / 2

            cumulative_loss += total_loss * batch_size
            num_samples += batch_size

            if i % 100 == 0:
                logger.info(
                    f"Test : [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                )

        val_metrics = get_clip_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        loss = cumulative_loss / num_samples
        metrics.update(
            {**val_metrics, "clip_test_loss": loss.item(), "num_samples": num_samples}
        )
    logger.info(
        f"Test: " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    for name, val in metrics.items():
        logger.info({f"test/{name}": val})

    return metrics


def main():
    args = create_args()
    set_random_seed(args.seed)
    # create logger
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    os.environ["WANDB_DIR"] = args.logging_dir
    
    if not args.wandb_id:
        args.wandb_id = wandb.util.generate_id()

    logger.add(os.path.join(args.logging_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_seed'+str(args.seed) + ".log"), level="INFO")
    
    logger.info(args)
    if args.close_wandb:
        os.environ["WANDB_DISABLED"]="true"

    wandb.init(
                project = "UrbanVLP",
                config = args,
                name = args.experiment_name,
                id = args.wandb_id,
                mode='dryrun',
                #resume = True,
                )
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_satellite, _, transform = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained='laion2b_s34b_b88k', 
        output_dict=True,
    )
    model_streetview, _, transform = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained='laion2b_s34b_b88k', 
        output_dict=True,
    )
    if args.gatefusion:
        model = MultiGranularity_GeoCLIP_GateFusion(
            model_satellite,
            model_streetview,
        )
    else:
        model = MultiGranularity_GeoCLIP(
            model_satellite,
            model_streetview,
        )
    model.to(device)
    
    logger.info('------------model----------')
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f'{total_params/(1024*1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f'{total_trainable_params:,} training parameters.')
    logger.info(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
    logger.info('------------model done----------')
    
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        args, transform, tokenizer
    )
    logger.info("train dataset size: {}".format(len(train_dataset)))
    logger.info("val dataset size: {}".format(len(val_dataset)))
    logger.info("test dataset size: {}".format(len(test_dataset)))
    # create dataloaders
    train_dataloader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                drop_last=True,
    )
    val_dataloader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                drop_last=False
    )
    test_dataloader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                drop_last=False
    )
    data = {}
    data["train_loader"] = train_dataloader
    data["val_loader"] = val_dataloader
    data["test_loader"] = test_dataloader

    # create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    criterion = open_clip.ClipLoss_filip()

    best_clip_val_loss = float("inf")
    for epoch in tqdm(range(args.epoch_num), desc="Training"):
        logger.info("Start epoch {}".format(epoch))

        train_one_epoch(
                            model,
                            criterion, 
                            data, 
                            epoch, 
                            optimizer, 
                            args, 
                            logger,
                        )
        completed_epoch = epoch + 1

        cur_metrics = evaluate(
                            model,
                            data, 
                            completed_epoch, 
                            args, 
                            logger,
                        )

        if cur_metrics["clip_val_loss"] < best_clip_val_loss:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "optimizer": optimizer.state_dict(),
                'model':model.state_dict()
            }
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_dir, f"best_states_epoch{completed_epoch}.pth"),
            )
            best_clip_val_loss = cur_metrics["clip_val_loss"]

if __name__ == "__main__":
    main()
