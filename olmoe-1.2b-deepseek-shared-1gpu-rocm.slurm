#!/bin/bash
#SBATCH --job-name=olmoe-1.2b-deepseek-shared-1gpu-rocm
#SBATCH --output=/home/morg/students/sagiahrac/repos/olmo-rocm/slurm_output/olmoe-1.2b-deepseek-shared-1gpu-rocm-%j.out
#SBATCH --error=/home/morg/students/sagiahrac/repos/olmo-rocm/slurm_output/olmoe-1.2b-deepseek-shared-1gpu-rocm-%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-210
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:amd:1
#SBATCH --time=5-00:00:00

export ROCM_HOME=/opt/rocm-6.4.3
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

cd /home/morg/students/sagiahrac/repos/olmo-rocm

source .venv/bin/activate

torchrun \
    --standalone \
    --nproc_per_node=1 \
    --master_port=29503 \
    scripts/train.py \
    configs/olmoe/ablations/olmoe-1.2b-deepseek-shared-2gpu.yaml \
    --wandb.name=olmoe-1.2b-a0.5b-deepseek-shared-${SLURM_JOB_ID}
