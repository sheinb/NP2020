#!/bin/bash

##SBATCH --account=dsheinbe

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 4 CPU core
#SBATCH -n 2

# load modules for TensorFlow
module load anaconda/3.5 cuda/10.0.130 cudnn/7.4 tensorflow/1.14.0_gpu_py36 anaconda/3-5.2.0

# Move to TF models/research directory
cd /gpfs/home/dsheinbe/tf/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/gpfs/home/dsheinbe/tf/dice/embedded_ssd_mobilenet_v1_coco_VOCckpt.config
TRAINED_CKPT_PREFIX=/gpfs/home/dsheinbe/tf/VOCdice/model.ckpt-35947
EXPORT_DIR=/gpfs/home/dsheinbe/tf/dice/export
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}

../../bin/bonnet_model_compiler.par \
--frozen_graph_path=${EXPORT_DIR}/frozen_inference_graph.pb \
--output_graph_path=${EXPORT_DIR}/dice_detector.binaryproto \
--input_tensor_name="Preprocessor/sub" \
--output_tensor_names="concat,concat_1" \
--input_tensor_size=256 \
--debug
