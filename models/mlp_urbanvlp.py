import math
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import datetime
import ast
import ipdb
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from utils import (
    LinearProbDataset_w_streetviewimg_w_coordinate,
    count_trainable_parameters,
    count_all_parameters,
    set_random_seed,
)
from tqdm import tqdm
from loguru import logger
import wandb
from models import GeoCLIP_LocationEncoder,MultiGranularity_GeoCLIP, MultiGranularity_GeoCLIP_GateFusion

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        choices=["Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
        help="which dataset",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./data/downstream_task/Beijing_test.csv",
        help="test file path, if None then only train and val",
    )
    parser.add_argument(
        "--linear_probe", 
        type=bool, 
        default=True, 
        help="training if True else testing"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        default="carbon",
        choices=["carbon", "population", "gdp", "poi", "nightlight",'houseprice'],
        help="indicator",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=3e-4, 
        help="learning rate"
    )
    parser.add_argument(
        "--wd", 
        type=float,
        default=0.01,
        help="weight decay"
    )
    parser.add_argument(
        "--drop_out", 
        type=float, 
        default=0.01, 
        help="dropout in linear probe",
    )
    parser.add_argument(
                "--batch_size", 
                type=int, 
                default=2, 
                help="batch size"
    )
    parser.add_argument(
        "--epoch_num", 
        type=int, 
        default=100, 
        help="epoch number"
    )
    parser.add_argument(
        "--log_every_n_steps", 
        type=int, 
        default=100, 
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        # default="./checkpoints/best_model.bin",
        default=None,
        help="pretrained model after running main.py",
    )
    parser.add_argument(
        "--img_embedding_dim", 
        type=int, 
        default=512,
        help="image encoder output dim"
    )
    parser.add_argument("--seed", type=int, default=132, help="random seed")
    parser.add_argument(
        "--logging_dir", type=str, default="logs/downtask1", help="logging directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/downtask1",
        help="checkpoint path",
    )
    # MLP parameters
    parser.add_argument(
        "--project_dim", 
        type=int, 
        default=256, 
        help="project dimension"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu"],
        help="activation function",
    )

    # data hyper-parameters
    parser.add_argument(
        "--train_dataset_ratio",
        type=float,
        default=0.8,
        help="ratio of training dataset",
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
        "--inference",
        action='store_true',
    )
    
    parser.add_argument(
        "--inference_model",
        type=str,
        default="checkpoints/downtask1",
        help="inference model for inference",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="downstream_carbon",
        # help="inference model for inference",
    )
    parser.add_argument(
        "--split_try", 
        type=int, 
        default=0, 
    )
    parser.add_argument(
        "--gatefusion",
        type=bool,
        default=False,
    )
    
    args = parser.parse_args()

    return args

class LinearProbe(nn.Module):
    def __init__(self, 
                 model,
                 args,
                 multiclass=False,
                ):
        super().__init__()
        self.model = model
        self.project = nn.Linear(args.img_embedding_dim, args.project_dim)
        self.activation = nn.ReLU() if args.activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(args.drop_out)
        if multiclass:
            #poi
            self.predict = nn.Linear(args.project_dim, 14)
        else:
            self.predict = nn.Linear(args.project_dim, 1)

    def forward(self, 
                images, 
                streetview_images,
                streetview_coordinates,
                ):
        image_latent_satellite = self.model.model_satellite.encode_image(images)
        image_latent_streetview = self.model.model_streetview.encode_image(streetview_images)
        image_latent = image_latent_satellite + image_latent_streetview
        
        bs = streetview_coordinates.shape[0]
        streetview_coordinates_embed_list = []
        for i in range(bs):
            streetview_coordinates_embed_list.append(self.model.gps_encoder(streetview_coordinates[i]))
        streetview_coordinates_embed = torch.stack(streetview_coordinates_embed_list)
        streetview_coordinates_embed = self.model.fc_layer(streetview_coordinates_embed.permute(0,2,1)).squeeze(-1)
        image_latent = image_latent + streetview_coordinates_embed
        
        image_latent = self.project(image_latent)
        image_latent = self.activation(image_latent)
        image_latent = self.dropout(image_latent)
        logits = self.predict(image_latent)
        return logits.squeeze(dim=1)
    
def create_datasets(args, transform):
    """To create train, val, test datasets."""
    if args.indicator == 'houseprice':
        if args.split_try == 0:
            data = pd.read_csv("data/downstream/Beijing_train_w_houseprice.csv")
        else:
            data = pd.read_csv(f"data/downstream_split_try{args.split_try}/Beijing_train_w_houseprice.csv")
    else:
        if args.split_try == 0:
            data = pd.read_csv(f"data/downstream/{args.dataset}_train.csv")
        else:
            data = pd.read_csv(f"data/downstream_split_try{args.split_try}/{args.dataset}_train.csv")
        if args.indicator == 'poi':
            data = data[data['poi'] != 'unknown']
        
    # split dataset into train, val, test
    data = data.sample(frac=1).reset_index(drop=True)
    train_data = data[: int(len(data) * args.train_dataset_ratio)].reset_index(
        drop=True
    )
    val_data = data[int(len(data) * args.train_dataset_ratio) :].reset_index(drop=True)
    if args.indicator != 'poi':
        mean = np.mean(train_data[args.indicator])
        std = np.std(train_data[args.indicator])
    else:
        poi_cates = ['Dining and Cuisine', 'Leisure and Entertainment', 'Sports and Fitness', 'Business and Residential', 'Healthcare' 'Financial Institutions', 'Tourist Attractions', 'Lifestyle Services', 'Shopping and Consumption', 'Automobile Related', 'Hotel Accommodation', 'Transport Facilities', 'Science, Education and Culture', 'Companies and Enterprises']
        class_info = {f'{i}': [] for i in poi_cates}
        for item in tqdm(train_data[args.indicator]):
            item_dict = ast.literal_eval(item)
            for key, value in item_dict.items():
                if key in class_info:
                    class_info[key].append(value)
        mean_dict = {}
        std_dict = {}
        for k,v in class_info.items():
            mean_dict[k] = np.mean(v)
            std_dict[k] = np.std(v)
        mean = mean_dict
        std = std_dict
        
    # create datasets
    train_dataset = LinearProbDataset_w_streetviewimg_w_coordinate(
        args.dataset, 
        train_data, 
        args.indicator, 
        transform, 
        mean, 
        std, 
        is_test=False,
    )
    
    val_dataset = LinearProbDataset_w_streetviewimg_w_coordinate(
        args.dataset, 
        val_data, 
        args.indicator, 
        transform, 
        mean, 
        std, 
        is_test=False
    )

    if args.test_file is not None:
        test_data = pd.read_csv(args.test_file)
        test_dataset = LinearProbDataset_w_streetviewimg_w_coordinate(
            args.dataset, 
            test_data, 
            args.indicator, 
            transform, 
            mean, 
            std, 
        )
        return train_dataset, val_dataset, test_dataset, mean, std
    else:
        return train_dataset, val_dataset, None, mean, std


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


def train_one_epoch(model, 
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
            y,
            streetview_images,
            streetview_coordinates,
        ) = batch  
        images = images.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        streetview_images = streetview_images.to(device=device, non_blocking=True)
        streetview_coordinates = streetview_coordinates.to(device=device, non_blocking=True)
        
        streetview_images_feat = model.model.fc_layer(streetview_images.permute(0,2,3,4,1)).squeeze(-1)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        predicts = model(images, 
                         streetview_images_feat,
                         streetview_coordinates,
                         )
        loss = criterion(predicts, y)
        wandb.log({
                        'train/loss':loss.item(), 
                        'epoch':epoch
                   })
        loss.backward()

        optimizer.step()

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
            print(f"y: {y}")
            print(f"predicts: {predicts}")
            for key in ["mse", "r2", "rmse", "mae", "mape"]:
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
            losses_m["mse"].update(loss.item(), batch_size)
            losses_m["r2"].update(
                r2_score(
                    y_true = y.cpu().numpy(), 
                    y_pred = predicts.detach().cpu().numpy(),
                ),
                
                batch_size
            )
            
            print(f"r2 score: {r2_score(y_true = y.cpu().numpy(), y_pred = predicts.detach().cpu().numpy())}")
            
            losses_m["rmse"].update(
                np.sqrt(
                    mean_squared_error(y.cpu().numpy(), predicts.detach().cpu().numpy())
                ),
                batch_size,
            )
            losses_m["mae"].update(
                mean_absolute_error(y.cpu().numpy(), predicts.detach().cpu().numpy()),
                batch_size,
            )
            losses_m["mape"].update(
                mean_absolute_percentage_error(
                    y.cpu().numpy(), predicts.detach().cpu().numpy()
                ),
                batch_size,
            )

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
                f"Metrics: " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                logger.info({name: val, "step": step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def evaluate(
                model, 
                data, 
                epoch, 
                args,
                logger
             ):
    """To evaluate on val dataset."""
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = data["val_loader"]
    all_y, all_predicts = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (
                images,
                y,
                streetview_images,
                streetview_coordinates,
            ) = batch  
            images = images.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            streetview_images = streetview_images.to(device=device, non_blocking=True)
            streetview_coordinates = streetview_coordinates.to(device=device, non_blocking=True)

            
            streetview_images_feat = model.model.fc_layer(streetview_images.permute(0,2,3,4,1)).squeeze(-1)

            
            y_hat = model(images, 
                         streetview_images_feat,
                         streetview_coordinates,
                         )

            all_y.append(y.cpu().numpy())
            all_predicts.append(y_hat.cpu().numpy())
    all_y = np.concatenate(all_y)
    all_predicts = np.concatenate(all_predicts)

    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    metrics = {}
    dataloader = data["test_loader"]
    all_y, all_predicts = [], []
    
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (
                images,
                y,
                streetview_images,
                streetview_coordinates,
            ) = batch  
            images = images.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            streetview_images = streetview_images.to(device=device, non_blocking=True)
            streetview_coordinates = streetview_coordinates.to(device=device, non_blocking=True)


            streetview_images_feat = model.model.fc_layer(streetview_images.permute(0,2,3,4,1)).squeeze(-1)
            y_hat = model(images, 
                          streetview_images_feat,
                          streetview_coordinates,
                          )

            all_y.append(y.cpu().numpy())
            all_predicts.append(y_hat.cpu().numpy())
    end_time = time.time()

    num_frames = len(dataloader) * args.batch_size
    total_time = end_time - start_time
    fps = num_frames / total_time
    logger.info(f"FPS: {fps:.2f}")

    all_y = np.concatenate(all_y)
    all_predicts = np.concatenate(all_predicts)

    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)
    logger.info(
        f"Test: " + "\t".join([f"{k}: {round(v, 3):.3f}" for k, v in metrics.items()])
    )

    #--------Output Result----------
    test_data = pd.read_csv(args.test_file)
    if args.indicator == 'poi':
        std_list = []
        mean_list = []
        for k,v in data['std'].items():
            std_list.append(v)
        for k,v in data['mean'].items():
            mean_list.append(v)
        y_hat = (all_predicts * std_list) + mean_list
        y_hat = [";".join(map(str, row)) for row in y_hat]
        
    else:
        y_hat = [item * data["std"] + data["mean"] for item in all_predicts]
    
    test_data[args.indicator + "_predict"] = y_hat
    test_data.to_csv(os.path.join(args.logging_dir,args.test_file.split('/')[2][:-4] + f"_predicted_{args.indicator}.csv"), index=False)


def main():
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
                name = f"{args.experiment_name}_lr{args.lr}_bs{args.batch_size}",
                id = args.wandb_id,
                mode='dryrun',
                #resume = True,
                )
    
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    clip_model_satellite, _, transform = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
    )
    clip_model_streetview, _, transform = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            
    )
    if args.gatefusion:
        model_pretrain = MultiGranularity_GeoCLIP_GateFusion(
                    clip_model_satellite,
                    clip_model_streetview,
        )
    else:
        model_pretrain = MultiGranularity_GeoCLIP(
                    clip_model_satellite,
                    clip_model_streetview,
        )
    pretrained_model_weight = torch.load(args.pretrained_model)
    model_pretrain.load_state_dict(pretrained_model_weight['model'])

    if args.indicator == 'poi':
        model = LinearProbe( 
                    model_pretrain,
                    args,
                    multiclass=True,
        )
    else:
        model = LinearProbe(
                            model_pretrain,
                            args
                )
    
    model.to(device)
    # import ipdb;ipdb.set_trace()
    for param in model.model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f'{total_params/(1024*1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'{total_trainable_params:,} training parameters.')
    logger.info(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
    logger.info('Trained model parts: ')
    for name, params in model.named_parameters():
        if params.requires_grad:
            logger.info(name)

    # create datasets
    train_dataset, val_dataset, test_dataset, mean, std = create_datasets(
        args, transform
    )
    logger.info("train dataset size: {}".format(len(train_dataset)))
    logger.info("val dataset size: {}".format(len(val_dataset)))
    logger.info("test dataset size: {}".format(len(test_dataset)))

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
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
        drop_last=False,
    )
    
    data = {}
    data["train_loader"] = train_dataloader
    data["val_loader"] = val_dataloader
    data["test_loader"] = test_dataloader
    data["mean"] = mean
    data["std"] = std

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    best_mse_val_loss = float("inf")
    for epoch in tqdm(range(args.epoch_num), desc="Training"):
        logger.info("Start epoch {}".format(epoch))

        train_one_epoch(model, 
                        criterion, 
                        data, 
                        epoch, 
                        optimizer, 
                        args, 
                        logger,
                        )
        completed_epoch = epoch + 1

        cur_metrics = evaluate(model, 
                               data, 
                               completed_epoch, 
                               args, 
                               logger,
                               )

        checkpoint_dict = {
            "epoch": completed_epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if cur_metrics["mse"] < best_mse_val_loss:
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_dir, f"best_states_epoch.pth"),
                
            )

            best_mse_val_loss = cur_metrics["mse"]

    load_path = os.path.join(args.checkpoint_dir, "best_states_epoch.pth")
    logger.info(f'load from {load_path}')
    best_checkpoint = torch.load(
        load_path,
        map_location=torch.device("cpu")
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    model.to(device)
    if args.test_file is not None:
        inference(model, data, args, logger)


def main_inference():
    set_random_seed(args.seed)
    # create logger
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    logger.add(
            os.path.join(args.logging_dir, 'inference_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_seed'+str(args.seed) + ".log"),
            level="INFO"
            )
    logger.info(args)
    
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model_satellite, _, transform = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
    )
    clip_model_streetview, _, transform = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",

    )
    model_pretrain = MultiGranularity_GeoCLIP(
                clip_model_satellite,
                clip_model_streetview,
    )
    if args.indicator == 'poi':
        model = LinearProbe(
                        model_pretrain,
                        args,
                        multiclass=True,
                )
    else:
        model = LinearProbe(
                    model_pretrain,
                    args
            )
    
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f'{total_params/(1024*1024):.2f}M total parameters.')

    # create datasets
    train_dataset, val_dataset, test_dataset, mean, std = create_datasets(
        args, transform
    )
    logger.info("train dataset size: {}".format(len(train_dataset)))
    logger.info("val dataset size: {}".format(len(val_dataset)))
    logger.info("test dataset size: {}".format(len(test_dataset)))

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
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
        drop_last=False,
    )
    
    data = {}
    data["train_loader"] = train_dataloader
    data["val_loader"] = val_dataloader
    data["test_loader"] = test_dataloader
    data["mean"] = mean
    data["std"] = std

    best_checkpoint = torch.load(
        args.inference_model, 
        map_location=torch.device("cpu")
    )
    logger.info(f"----------Load checkpoint from epoch {best_checkpoint['epoch']}!---------")
    
    model.load_state_dict(best_checkpoint['state_dict'])
    model.to(device)
    if args.test_file is not None:
        inference(model, data, args, logger)

if __name__ == "__main__":
    args = create_args()
    if args.inference:
        main_inference()
    else:
        main()
