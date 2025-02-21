import json
import random
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pynvml
from loguru import logger
# import ipdb

def count_trainable_parameters(model):
    """To compute the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """To compute the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def set_random_seed(seed):
    """To set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def get_clip_metrics(image_features, text_features, logit_scale, prefix=''):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {f"{prefix}_image_to_text": logits_per_image, 
              f"{prefix}_text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


class MultigranuDataset_full_preload(Dataset):
    def __init__(self, 
                        list_data=None, 
                        transform=None, 
                        tokenizer=None,
                        return_coordinate = False,
                        data_path='./',
                        city = 'Beijing',
                 ):
        super().__init__()

        self.transform = transform  # image transform for CoCa
        self.tokenizer = tokenizer  # tokenizer for CoCa
        self.return_coordinate = return_coordinate
        streetview_img_tensors_dict = torch.load(f'streetview_img_tensors_dict_{city}.npy')
        
        streetview_descriptions_dict = torch.load(f'streetview_descriptions_complexprompt_dict_{city}.npy')
        streetview_coordinates_dict = torch.load(f'streetview_coordinates_dict_{city}.npy')
        assert len(streetview_img_tensors_dict) == len(streetview_descriptions_dict) \
            and len(streetview_img_tensors_dict) == len(streetview_coordinates_dict)

        self.img_tensors = []
        self.captions = []
        self.caption_tokens = []
        self.streetview_img_tensors = []
        self.streetview_descriptions = []
        self.streetview_coordinates = []
        print('-----------Dataset initializing-------------')
        for item in tqdm(list_data):
            _index = np.random.randint(
                0, len(item)
            )  # randomly select one caption for each image
            self.captions.append(item[_index]["caption"])
            satellite_img_path = item[_index]["image"]
            img = Image.open(
                os.path.join(data_path, "urbansat", "satellite_images", item[_index]["image"])
            ).convert("RGB")
            img = transform(img)  # [3, 224, 224]
            self.img_tensors.append(img)
            self.caption_tokens.append(
                self.tokenizer(item[_index]["caption"])
            )  # [1, 77]
            self.streetview_img_tensors.append(streetview_img_tensors_dict[satellite_img_path])
            self.streetview_descriptions.append(streetview_descriptions_dict[satellite_img_path])
            self.streetview_coordinates.append(streetview_coordinates_dict[satellite_img_path])            

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        if len(self.streetview_img_tensors[index]) == 0:
            streetview_img_tensor = torch.zeros(25, 3, 224,224)
            streetview_coordinates = torch.zeros(25,2)
        else:
            if len(self.streetview_img_tensors[index]) > 25:
                self.streetview_img_tensors[index] = self.streetview_img_tensors[index][:25]
                self.streetview_coordinates[index] = self.streetview_coordinates[index][:25]
            streetview_img_tensor = torch.stack(self.streetview_img_tensors[index], dim=0)
            zeros_img = torch.zeros(25-len(self.streetview_img_tensors[index]), 3, 224,224)
            streetview_img_tensor = torch.cat((streetview_img_tensor, zeros_img), dim=0)
            
            streetview_coordinates = torch.tensor(self.streetview_coordinates[index])
            zeros_coordinates = torch.zeros(25-len(self.streetview_img_tensors[index]), 2)
            streetview_coordinates = torch.cat((streetview_coordinates, zeros_coordinates), dim=0)
            
        if self.return_coordinate:
            return self.img_tensors[index], self.caption_tokens[index], streetview_img_tensor, self.streetview_descriptions[index], streetview_coordinates
        else:
            return self.img_tensors[index], self.caption_tokens[index], streetview_img_tensor, self.streetview_descriptions[index]


class MultigranuDataset(Dataset):
    def __init__(self, list_data=None, transform=None, tokenizer=None):
        super().__init__()

        self.transform = transform  
        self.tokenizer = tokenizer  

        self.img_paths = []
        self.img_tensors = []
        self.streetview_img_paths = []
        self.captions = []
        self.caption_tokens = []
        print('-----------Dataset initializing-------------')
        for item in tqdm(list_data):
            _index = np.random.randint(
                0, len(item)
            )
            self.img_paths.append(os.path.join("./data/satellite_images", item[_index]["image"]))
            self.captions.append(item[_index]["caption"])
            im = Image.open(
                os.path.join("./data/satellite_images", item[_index]["image"])
            ).convert("RGB")
            im = transform(im)  
            self.img_tensors.append(im)
            self.caption_tokens.append(
                self.tokenizer(item[_index]["caption"])
            )
            
            streetview_img_corresponding_path = f"data/streetview_corresponding/Beijing/text/{item[_index]['image'].split('/')[1].replace('jpg','txt')}"
            streetview_img_names = []
            if os.path.exists(streetview_img_corresponding_path):
                # continue
                with open(streetview_img_corresponding_path,'r')as f:
                    for line in f:
                        streetview_img_names.append(line.replace('\n',''))
            self.streetview_img_paths.append(streetview_img_names)
            

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        streetview_img_tensor_list = []
        if len(self.streetview_img_paths[index]) > 0:
            for streetview_img_name in self.streetview_img_paths[index]:
                streetview_img = Image.open(
                    os.path.join("./data/streetview_images", streetview_img_name)
                ).convert("RGB")
                streetview_img = self.transform(streetview_img)  # [3, 224, 224]
                streetview_img_tensor_list.append(streetview_img)
        # self.streetview_img_tensors.append(streetview_img_list)
        if len(streetview_img_tensor_list) == 0:
            streetview_img_tensor = torch.zeros(25, 3, 224,224)
        else:
            streetview_img_tensor = torch.stack(streetview_img_tensor_list, dim=0)
            
            zeros = torch.zeros(25-len(streetview_img_tensor_list), 3, 224,224)
            streetview_img_tensor = torch.cat((streetview_img_tensor,zeros), dim=0)
            
        return self.img_tensors[index], self.caption_tokens[index], streetview_img_tensor

class LinearProbDataset(Dataset):
    """Dataset for linear probe task.

    Args:
        data_name (str): name of dataset, Beijing or Shanghai
        df_data (DataFrame): dataframe of data
        indicator (str): indicator to predict, CO2, O3, SO2
        transform (torchvision.transforms): image transform for CoCa
        mean (float): mean of indicator values
        std (float): std of indicator values
        is_test (bool): whether this is test set
    """

    def __init__(
        self,
        data_name="Beijing",
        df_data=None,
        indicator="CO2",
        transform=None,
        mean=1.0,
        std=1.0,
        is_test=False,
    ):
        super().__init__()

        self.transform = transform  # image transform for CoCa
        
        self.indicator = indicator
        self.img_tensors = []
        self.y = []
        if indicator == 'poi':
            mean_list = [v for k,v in mean.items()]
            std_list = [v for k,v in std.items()]
        for idx, row in df_data.iterrows():
            _coordinate = eval(row["Coordinate"])  # tuple
            _image_name = "16_{}_{}_s.jpg".format(_coordinate[0], _coordinate[1])
            if data_name == "Beijing":
                _image_path = os.path.join("./data/satellite_images/Beijing", _image_name)
            elif data_name == "Shanghai":
                _image_path = os.path.join("./data/satellite_images/Shanghai", _image_name)
            elif data_name == "Shenzhen":
                _image_path = os.path.join("./data/satellite_images/Shenzhen", _image_name)
            elif data_name == "Guangzhou":
                _image_path = os.path.join("./data/satellite_images/Guangzhou", _image_name)
            else:
                raise ValueError("data must be Beijing or Shanghai")

            _im = Image.open(_image_path).convert("RGB")

            _im = transform(_im)  # [3, 224, 224]
            self.img_tensors.append(_im)

            if indicator == 'poi':
                l_ = []
                for i in row[indicator][1:-1].split(', '):
                    # print(i)
                    l_.append(int(i.split(':')[1]))
                normalized_l_ = [(ll - mean_) / std_ for ll, mean_, std_ in zip(l_, mean_list, std_list)]

                self.y.append(normalized_l_)
                
            else:
                self.y.append((row[indicator] - mean) / std)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.img_tensors[index], np.float32(self.y[index])

class LinearProbDataset_w_streetviewimg(Dataset):
    """Dataset for linear probe task.

    Args:
        data_name (str): name of dataset, Beijing or Shanghai
        df_data (DataFrame): dataframe of data
        indicator (str): indicator to predict, CO2, O3, SO2
        transform (torchvision.transforms): image transform for CoCa
        mean (float): mean of indicator values
        std (float): std of indicator values
        is_test (bool): whether this is test set
    """

    def __init__(
        self,
        data_name="Beijing",
        df_data=None,
        indicator="CO2",
        transform=None,
        mean=1.0,
        std=1.0,
        is_test=False,
    ):
        super().__init__()

        self.transform = transform  # image transform for CoCa

        # self.img_paths = []
        self.img_tensors = []
        self.y = []
        
        self.streetview_img_tensors_dict = torch.load(f'streetview_img_tensors_dict_{data_name}.npy')
        self.streetview_img_tensors = []
        
        for idx, row in df_data.iterrows():
            _coordinate = eval(row["Coordinate"])  # tuple
            _image_name = "16_{}_{}_s.jpg".format(_coordinate[0], _coordinate[1])
            if data_name == "Beijing":
                _image_path = os.path.join("./data/satellite_images/Beijing", _image_name)
            elif data_name == "Shanghai":
                _image_path = os.path.join("./data/satellite_images/Shanghai", _image_name)
            else:
                raise ValueError("data must be Beijing or Shanghai")

            _im = Image.open(_image_path).convert("RGB")
            _im = transform(_im)  # [3, 224, 224]
            self.img_tensors.append(_im)
            self.streetview_img_tensors.append(self.streetview_img_tensors_dict[f"Beijing/{_image_name}"])
            
            if indicator == 'poi':
                l_ = []
                for i in row[indicator][1:-1].split(', '):
                    l_.append(int(i.split(':')[1]))
                self.y.append(l_)
            else:
                self.y.append((row[indicator] - mean) / std)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if len(self.streetview_img_tensors[index]) == 0:
            streetview_img_tensor = torch.zeros(25, 3, 224,224)
        else:
            streetview_img_tensor = torch.stack(self.streetview_img_tensors[index], dim=0)
            
            zeros = torch.zeros(25-len(self.streetview_img_tensors[index]), 3, 224,224)
            streetview_img_tensor = torch.cat((streetview_img_tensor,zeros), dim=0)
        
        return self.img_tensors[index], np.float32(self.y[index]), streetview_img_tensor

class LinearProbDataset_w_streetviewimg_w_coordinate(Dataset):
    """Dataset for linear probe task.

    Args:
        data_name (str): name of dataset, Beijing or Shanghai
        df_data (DataFrame): dataframe of data
        indicator (str): indicator to predict, CO2, O3, SO2
        transform (torchvision.transforms): image transform for CoCa
        mean (float): mean of indicator values
        std (float): std of indicator values
        is_test (bool): whether this is test set
    """

    def __init__(
        self,
        data_name="Beijing",
        df_data=None,
        indicator="CO2",
        transform=None,
        mean=1.0,
        std=1.0,
        is_test=False,
    ):
        super().__init__()

        self.transform = transform  

        self.img_tensors = []
        self.y = []
        self.streetview_img_tensors_dict = torch.load(f'streetview_img_tensors_dict_{data_name}.npy')
        self.streetview_coordinates_dict = torch.load(f'streetview_coordinates_dict_{data_name}.npy')
        
        self.streetview_img_tensors = []
        self.streetview_coordinates = []
        for idx, row in df_data.iterrows():
            _coordinate = eval(row["Coordinate"])  # tuple
            _image_name = "16_{}_{}_s.jpg".format(_coordinate[0], _coordinate[1])
            _image_path = os.path.join(f"./data/urbansat/satellite_images/{data_name}", _image_name)

            _im = Image.open(_image_path).convert("RGB")
            _im = transform(_im)  # [3, 224, 224]
            self.img_tensors.append(_im)
            self.streetview_img_tensors.append(self.streetview_img_tensors_dict[f"{data_name}/{_image_name}"])
            self.streetview_coordinates.append(self.streetview_coordinates_dict[f"{data_name}/{_image_name}"])

            if indicator == 'poi':
                mean_list = [v for k,v in mean.items()]
                std_list = [v for k,v in std.items()]
                l_ = []
                for i in row[indicator][1:-1].split(', '):
                    l_.append(int(i.split(':')[1]))
                normalized_l_ = [(ll - mean_) / std_ for ll, mean_, std_ in zip(l_, mean_list, std_list)]
                self.y.append(normalized_l_)
            else:
                self.y.append((row[indicator] - mean) / std)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if len(self.streetview_img_tensors[index]) == 0:
            streetview_img_tensor = torch.zeros(25, 3, 224,224)
            streetview_coordinates = torch.zeros(25,2)
        else:
            if len(self.streetview_img_tensors[index]) > 25:
                self.streetview_img_tensors[index] = self.streetview_img_tensors[index][:25]
                self.streetview_coordinates[index] = self.streetview_coordinates[index][:25]
            streetview_img_tensor = torch.stack(self.streetview_img_tensors[index], dim=0)

            zeros_img = torch.zeros(25-len(self.streetview_img_tensors[index]), 3, 224,224)
            streetview_img_tensor = torch.cat((streetview_img_tensor, zeros_img), dim=0)
            
            streetview_coordinates = torch.tensor(self.streetview_coordinates[index])
            zeros_coordinates = torch.zeros(25-len(self.streetview_img_tensors[index]), 2)
            streetview_coordinates = torch.cat((streetview_coordinates, zeros_coordinates), dim=0)
            
        return self.img_tensors[index], np.float32(self.y[index]), streetview_img_tensor, streetview_coordinates


class GenerationDataset(Dataset):
    """Dataset for text generation task.

    Args:
        data_name (str): name of dataset, Beijing or Shanghai
        df_data (DataFrame): dataframe of data
        indicator (str): indicator to predict, CO2, O3, SO2
        transform (torchvision.transforms): image transform for CoCa
        mean (float): mean of indicator values
        std (float): std of indicator values
        is_test (bool): whether this is test set
    """

    def __init__(
        self,
        jpg_list=None,
        transform=None,
    ):
        super().__init__()

        self.jpg_list = jpg_list
        self.transform = transform 
        self.img_tensors = []
        for jpg_path in jpg_list:
            _im = Image.open(str(jpg_path)).convert("RGB")
            _im = transform(_im)
            self.img_tensors.append(_im)

    def __len__(self):
        return len(self.img_tensors)

    def __getitem__(self, index):
        return self.img_tensors[index]


def get_GPU_usage():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(f"GPU {i}: Total memory: {info.total/1024**3} GB, Used memory: {info.used/1024**3} GB, Memory utilization: {(info.used/info.total)*100:.2f}%")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    print()
