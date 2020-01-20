#!/bin/bash

##SBATCH --account=dsheinbe

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:2

# Request 4 CPU core
#SBATCH -n 4

#SBATCH -t 6:00:00

# load modules for TensorFlow
module load cuda/10.0.130 cudnn/7.4 tensorflow/1.14.0_gpu_py36 anaconda/3-5.2.0

# Move to TF models/research directory
cd /gpfs/home/dsheinbe/tf/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#PIPELINE_CONFIG_PATH=/gpfs/home/dsheinbe/tf/dice/embedded_ssd_mobilenet_v1_coco.config
#PIPELINE_CONFIG_PATH=/gpfs/home/dsheinbe/tf/dice/embedded_ssd_mobilenet_v1_coco_ckpt.config
PIPELINE_CONFIG_PATH=/gpfs/home/dsheinbe/tf/dice/embedded_ssd_mobilenet_v1_coco_VOCckpt.config
MODEL_DIR=/gpfs/home/dsheinbe/tf/VOCdice
NUM_TRAIN_STEPS=130000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
