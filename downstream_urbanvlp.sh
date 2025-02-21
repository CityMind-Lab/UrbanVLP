datename=$(date +%Y%m%d-%H%M%S)
indicator=carbon
# indicator=population
# indicator=gdp
# indicator=poi
# indicator=houseprice
# indicator=nightlight
city=Beijing
# city=Shanghai
# city=Guangzhou
# city=Shenzhen
lr=3e-4
bs=128
model_name=urbanvlp
experiment_name=downstream_$indicator\_$model_name\_$city\_pretrainedbs32lr3e-6_refined_bs$bs\_lr$lr
pretrained_model=#
logging_dir=logs_$model_name\_$city/downstreamtask/$indicator/$experiment_name/$datename
mkdir -p $logging_dir
cp $0 $logging_dir/$(basename $0 .sh).sh
pip list > $logging_dir/environment.txt
cp -r models $logging_dir/models
HF_ENDPOINT=https://hf-mirror.com \
    python models/mlp_urbanvlp.py \
        --indicator $indicator \
        --dataset $city \
        --test_file data/downstream/$city\_test.csv \
        --lr $lr \
        --batch_size $bs \
        --experiment_name $experiment_name \
        --logging_dir $logging_dir \
        --checkpoint_dir $logging_dir/checkpoints \
        --pretrained_model $pretrained_model \
        --epoch_num 60 \
