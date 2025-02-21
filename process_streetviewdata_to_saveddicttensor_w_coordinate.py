import json
import numpy as np
import ipdb
from tqdm import tqdm
import os
from PIL import Image
import open_clip_mine as open_clip
import torch
from models.models import GeoCLIP_LocationEncoder


city = 'Beijing'
# city = 'Shanghai'
# city = 'Shenzhen'
# city = 'Guangzhou'
data = json.load(open(f"data/satellite_captions/{city}_captions.json", "r"))

model_satellite, _, transform = open_clip.create_model_and_transforms(
    model_name="ViT-B-16",
    pretrained='laion2b_s34b_b88k', 
    output_dict=True,
)
tokenizer = open_clip.get_tokenizer("ViT-L-14")

gps_encoder = GeoCLIP_LocationEncoder()
gps_encoder.load_state_dict(
        torch.load('location_encoder_weights.pth')
)
gps_encoder.cuda() 

img_paths = []
img_tensors = []
streetview_img_tensors = {}
streetview_coordinates = {}
streetview_coordinates_tensors = {}
streetview_coordinates_tensors_geoclip = {}
streetview_descriptions_tensors = {}
streetview_descriptions = {}
captions = []
caption_tokens = []
        
print('-----------Dataset initializing-------------')
for item in tqdm(data):
    _index = np.random.randint(
        0, len(item)
    )

    img_paths.append(os.path.join(f"./data/satellite_images/{city}", item[_index]["image"]))
    captions.append(item[_index]["caption"])
    im = Image.open(
        os.path.join("./data/satellite_images", item[_index]["image"])
    ).convert("RGB")
    # im = transform(im).unsqueeze(0)  # [1, 3, 224, 224]
    im = transform(im)  # [3, 224, 224]
    img_tensors.append(im)
    caption_tokens.append(
        tokenizer(item[_index]["caption"])
    )
    
    
    streetview_img_path = f"data/streetview_corresponding/{city}/text/{item[_index]['image'].split('/')[1].replace('jpg','txt')}"
    streetview_img_tensor_list = []
    streetview_description_tensor_list = []
    streetview_description_list = []
    streetview_coordinate_list = []
    streetview_coordinate_tensor_list = []
    streetview_coordinate_tensor_geoclip_list = []
    if os.path.exists(streetview_img_path):
        streetview_img_names = []
        with open(streetview_img_path,'r')as f:
            for line in f:
                if 'download' in line:
                    continue
                streetview_img_names.append(line.replace('\n',''))
                
        if len(streetview_img_names) > 0:
            for streetview_img_name in streetview_img_names:
                print(streetview_img_name)
                streetview_img = Image.open(
                    os.path.join(f"./data/streetview_images/{city}", streetview_img_name)
                ).convert("RGB")
                streetview_img = transform(streetview_img)  # [3, 224, 224]
                streetview_img_tensor_list.append(streetview_img)
                
                
                a = json.load(open(f"data/streetview_descriptions/{city}/captions/{streetview_img_name.replace('jpg','json')}", "r"))   
                
                desc = list(a.values())[0]

                streetview_description_tensor_list.append(tokenizer(desc))
                streetview_description_list.append(desc)
                
                streetview_coor = [float(x) for x in streetview_img_name.split('_')[1].split(',')[::-1]]
                streetview_coordinate_list.append(streetview_coor)
                
                streetview_coor_tensor = torch.tensor(streetview_coor).unsqueeze(dim=0)
                streetview_coor_tensor = streetview_coor_tensor.cuda()
                streetview_coordinate_tensor_list.append(streetview_coor_tensor)
 
                streetview_coor_geoclip_tensor = gps_encoder(streetview_coor_tensor)
                streetview_coordinate_tensor_geoclip_list.append(streetview_coor_geoclip_tensor)

    
    streetview_img_tensors[item[_index]["image"]] = streetview_img_tensor_list
    streetview_descriptions_tensors[item[_index]["image"]] = streetview_description_tensor_list
    streetview_descriptions[item[_index]["image"]] = tokenizer(' '.join(streetview_description_list))
    streetview_coordinates[item[_index]["image"]] = streetview_coordinate_list
    streetview_coordinates_tensors[item[_index]["image"]] = streetview_coordinate_tensor_list
    
    streetview_coordinates_tensors_geoclip[item[_index]["image"]] = streetview_coordinate_tensor_geoclip_list

torch.save(streetview_descriptions_tensors, f'streetview_descriptions_complexprompt_tensors_dict_{city}.npy')
torch.save(streetview_descriptions, f'streetview_descriptions_complexprompt_dict_{city}.npy')

torch.save(streetview_coordinates, f'streetview_coordinates_dict_{city}.npy')
torch.save(streetview_coordinates_tensors, f'streetview_coordinates_tensors_dict_{city}.npy')
torch.save(streetview_coordinates_tensors_geoclip, f'streetview_coordinates_tensors_geoclip_dict_{city}.npy')

torch.save(streetview_img_tensors, f'streetview_img_tensors_dict_{city}.npy')
