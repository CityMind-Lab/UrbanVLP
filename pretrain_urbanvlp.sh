datename=$(date +%Y%m%d-%H%M%S)
lr=3e-6
bs=32
city=Beijing
# city=Shanghai
# city=Guangzhou
# city=Shenzhen
experiment_name=urbanvlp_$city\_bs$bs\_lr$lr
logging_dir=logs_urbanvlp_$city/$experiment_name/$datename
mkdir -p $logging_dir
cp $0 $logging_dir/$(basename $0 .sh).sh
pip list > $logging_dir/environment.txt
cp -r models $logging_dir/models
HF_ENDPOINT=https://hf-mirror.com \
    python models/main_urbanvlp.py \
        --dataset $city \
        --data_path ./data \
        --lr $lr \
        --experiment_name $experiment_name \
        --logging_dir $logging_dir \
        --checkpoint_dir $logging_dir/checkpoints_$experiment_name \
        --batch_size $bs \
        --epoch_num 3 \
