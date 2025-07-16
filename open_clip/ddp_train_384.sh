#!/bin/bash
#SBATCH --job-name=geolb_training
#SBATCH --nodes=1
#SBATCH --gpus=6
#SBATCH --ntasks=6
#SBATCH --nodelist=fwgegpu06
#SBATCH --output=geolb_training.out
#SBATCH --time=10-8:0:0
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=150G


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WORLD_SIZE=6
export OMP_NUM_THREADS=16
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Dynamically assign a random port to avoid conflicts
export MASTER_PORT=$((12000 + RANDOM % 1000))

# Set the master node address
export MASTER_ADDR=localhost
export NCCL_P2P_LEVEL=NVL
# request problem: requests == 2.27.1 & set this 
export CURL_CA_BUNDLE=""

data_root=/home/yu34/GeoLangBind/data/

SINGULARITY_IMAGE=/home/yu34/GeoLangBind/pytorch-video-docker_2.5.1-cu118-20250101.sif

module load singularity

export PYTHONPATH="$PYTHONPATH:$PWD/src"
# Set the training arguments
singularity exec --nv $SINGULARITY_IMAGE torchrun --nnodes=1 --nproc_per_node=$WORLD_SIZE --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m src.open_clip_train.main \
  --batch-size 35 \
  --train-data $data_root \
  --precision amp \
  --log-every-n-steps 50 \
  --workers 4 \
  --report-to tensorboard \
  --save-frequency 1 \
  --dataset-type geolb \
  --warmup 1000 \
  --lr=5e-4 \
  --use-bn-sync \
  --wd=1e-7 \
  --epochs=50 \
  --lock-text \
  --siglip \
  --DOFA \
  --model GeoLB-ViT-14-so400m-SigLIP-384 \
  --pretrained webli