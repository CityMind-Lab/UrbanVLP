
# UrbanVLP
Official Implementation of AAAI2025 "UrbanVLP: Multi-Granularity Vision-Language Pretraining for Urban Socioeconomic Indicator Prediction".



## Install

Please refer to requirements.txt for relevent environment version.



## Data
First prepare dataset into `data/` ,we provide some examples for reference.

Satellite imagery can be collected through https://github.com/siruzhong/UrbanCLIP-Dataset-Toolkit.

Streetview Imagery can be collected from [Baidu Map](https://lbsyun.baidu.com/).

For downstream data please refer to data sources in our paper and retrieve the data based on the center point coordinates of the satellite imagery.

We preprocess the dataset to save time in training.
```
python process_streetviewdata_to_saveddicttensor_w_coordinate.py
```


## Pretrain
```
bash pretrain_urbanvlp.sh
```

## Finetune
Fill the path of the pretrained model weights into the `pretrained_model=`
```
bash downstream_urbanvlp.sh
```