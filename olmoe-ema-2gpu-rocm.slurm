#!/bin/bash
#SBATCH --job-name=olmoe-ema-2gpu-rocm
#SBATCH --output=/home/morg/students/sagiahrac/repos/olmo-rocm/slurm_output/olmoe-ema-2gpu-rocm-%j.out
#SBATCH --error=/home/morg/students/sagiahrac/repos/olmo-rocm/slurm_output/olmoe-ema-2gpu-rocm-%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=5-00:00:00

export ROCM_HOME=/opt/rocm-6.4.3

cd /home/morg/students/sagiahrac/repos/olmo-rocm

source .venv/bin/activate

torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    scripts/train.py \
    configs/olmoe/ablations/olmoe17-8x1b-local-ema-2gpu.yaml
