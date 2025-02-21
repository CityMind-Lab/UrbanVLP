import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'models'))  

from GeoCLIP_rff.layers import GaussianEncoding

class MGC(nn.Module):
    def __init__(self, 
                        clip_model_satellite, 
                        clip_model_streetview, 
                        # fc_layer,
                        attn_embed_dim=768,
                        num_heads = 8,
                        # args,
                ):
        super().__init__()
        self.model_satellite = clip_model_satellite
        self.model_streetview = clip_model_streetview
        self.fc_layer = nn.Linear(25,1)
        
        self.visual_crossattn_layer = nn.MultiheadAttention(
            embed_dim = attn_embed_dim, 
            num_heads = num_heads,
            batch_first = True
        )
        # sentence local attention layer
        self.text_crossattn_layer = nn.MultiheadAttention(
            embed_dim = attn_embed_dim,
            num_heads = num_heads,
            batch_first=True
        )

    def forward(self,             
                images,
                texts,
                streetview_images,
                streetview_texts,
                ):
        
        satellite_model_out = self.model_satellite(images, texts)


        
        # ipdb.set_trace()
        streetview_images = self.fc_layer(streetview_images.permute(0,2,3,4,1)).squeeze(-1)

        streetview_model_out = self.model_streetview(streetview_images, streetview_texts)
        satellite_model_out['image_all_tokens'] = satellite_model_out['image_all_tokens'] + streetview_model_out['image_all_tokens']
        
        satellite_visual_atten_output, _ = self.visual_crossattn_layer(
                    query = satellite_model_out['image_all_tokens'], 
                    key = satellite_model_out['text_all_tokens'], 
                    value = satellite_model_out['text_all_tokens'], 
                    # key_padding_mask=mask,
        )
        satellite_text_atten_output, _ = self.text_crossattn_layer(
                    query = satellite_model_out['text_all_tokens'], 
                    key = satellite_model_out['image_all_tokens'], 
                    value = satellite_model_out['image_all_tokens'], 
                    # key_padding_mask=mask,
        )
        streetview_visual_atten_output, _ = self.visual_crossattn_layer(
                    query = streetview_model_out['image_all_tokens'], 
                    key = streetview_model_out['text_all_tokens'], 
                    value = streetview_model_out['text_all_tokens'], 
                    # key_padding_mask=mask,
        )
        streetview_text_atten_output, _ = self.text_crossattn_layer(
                    query = streetview_model_out['text_all_tokens'], 
                    key = streetview_model_out['image_all_tokens'], 
                    value = streetview_model_out['image_all_tokens'], 
                    # key_padding_mask=mask,
        )
        streetview_model_out['streetview_visual_atten_output'] = streetview_visual_atten_output
        streetview_model_out['streetview_text_atten_output'] = streetview_text_atten_output
        satellite_model_out['satellite_visual_atten_output'] = satellite_visual_atten_output
        satellite_model_out['satellite_text_atten_output'] = satellite_text_atten_output
        
        
        return satellite_model_out, streetview_model_out

class MultiGranularity_GeoCLIP(nn.Module):
    def __init__(self, 
                        clip_model_satellite, 
                        clip_model_streetview, 
                ):
        super().__init__()
        self.model_satellite = clip_model_satellite
        self.model_streetview = clip_model_streetview
        self.fc_layer = nn.Linear(25,1)
        self.coordinate_fc_layer = nn.Linear(512,768)
        self.gps_encoder = GeoCLIP_LocationEncoder()
        self.gps_encoder.load_state_dict(
                torch.load('location_encoder_weights.pth')
        )

    def forward(self,             
                images,
                texts,
                streetview_images,
                streetview_texts,
                streetview_coordinates,
                ):
        
        satellite_model_out = self.model_satellite(images, texts)


        streetview_images = self.fc_layer(streetview_images.permute(0,2,3,4,1)).squeeze(-1)
        streetview_model_out = self.model_streetview(streetview_images, streetview_texts)
        satellite_model_out['image_features'] = satellite_model_out['image_features'] + streetview_model_out['image_features']

        
        bs = streetview_coordinates.shape[0]
        streetview_coordinates_embed_list = []
        for i in range(bs):
            streetview_coordinates_embed_list.append(self.gps_encoder(streetview_coordinates[i]))
        streetview_coordinates_embed = torch.stack(streetview_coordinates_embed_list)
        streetview_coordinates_embed = self.fc_layer(streetview_coordinates_embed.permute(0,2,1)).squeeze(-1)
        streetview_coordinates_embed = self.coordinate_fc_layer(streetview_coordinates_embed)
      
        
        satellite_model_out['image_features'] = satellite_model_out['image_features'] + streetview_coordinates_embed
        
        return satellite_model_out, streetview_model_out
    
    
class MultiGranularity_Structual(nn.Module):
    def __init__(self, 
                        clip_model_satellite, 
                        clip_model_streetview, 

                ):
        super().__init__()
        self.model_satellite = clip_model_satellite
        self.model_streetview = clip_model_streetview
        self.fc_layer = nn.Linear(25,1)
        self.coordinate_fc_layer = nn.Linear(512,768)
        self.gps_encoder = GeoCLIP_LocationEncoder()
        self.gps_encoder.load_state_dict(
                torch.load('location_encoder_weights.pth')
        )

    def forward(self,             
                images,
                texts,
                streetview_images,
                streetview_texts,
                streetview_coordinates,
                ):
        bs = streetview_images.shape[0]
        
        satellite_model_out = self.model_satellite(images, texts)

        
        streetview_images = streetview_images.mean(dim=1) + self.fc_layer(streetview_images.permute(0,2,3,4,1)).squeeze(-1)
        streetview_model_out = self.model_streetview(streetview_images, streetview_texts)
        beta = torch.softmax(streetview_model_out['image_features'], dim=1)                 # (M, 1)
        
        streetview_model_out['image_features'] = beta * streetview_model_out['image_features']
        
        satellite_model_out['image_features'] = satellite_model_out['image_features'] + streetview_model_out['image_features']
        
        return satellite_model_out, streetview_model_out
    
class Streetview_GeoCLIP(nn.Module):
    def __init__(self, 
                        clip_model_streetview, 
                ):
        super().__init__()
        # self.model_satellite = clip_model_satellite
        self.model_streetview = clip_model_streetview
        self.fc_layer = nn.Linear(25,1)
        self.coordinate_fc_layer = nn.Linear(512,768)
        self.gps_encoder = GeoCLIP_LocationEncoder()
        self.gps_encoder.load_state_dict(
                torch.load('location_encoder_weights.pth')
        )

    def forward(self,             
                # images,
                # texts,
                streetview_images,
                streetview_texts,
                streetview_coordinates,
                ):
        
        streetview_images = self.fc_layer(streetview_images.permute(0,2,3,4,1)).squeeze(-1)
        streetview_model_out = self.model_streetview(streetview_images, streetview_texts)

        
        
        bs = streetview_coordinates.shape[0]
        streetview_coordinates_embed_list = []
        for i in range(bs):
            streetview_coordinates_embed_list.append(self.gps_encoder(streetview_coordinates[i]))
        streetview_coordinates_embed = torch.stack(streetview_coordinates_embed_list)
        # import ipdb;ipdb.set_trace()
        streetview_coordinates_embed = self.fc_layer(streetview_coordinates_embed.permute(0,2,1)).squeeze(-1)
        streetview_coordinates_embed = self.coordinate_fc_layer(streetview_coordinates_embed)
      
        
        # satellite_model_out['image_features'] = satellite_model_out['image_features'] + streetview_coordinates_embed
        streetview_model_out['image_features'] = streetview_model_out['image_features'] + streetview_coordinates_embed
        return streetview_model_out

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180



class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class GeoCLIP_LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super().__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

    def forward(self, location):
        location = location.float()
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], 512).to('cuda')

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        
        return location_features
