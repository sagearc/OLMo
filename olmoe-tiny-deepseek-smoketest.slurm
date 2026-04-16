#!/bin/bash
#SBATCH --job-name=olmoe-tiny-deepseek-smoketest
#SBATCH --output=/home/morg/students/sagiahrac/repos/olmo-rocm/slurm_output/olmoe-tiny-deepseek-smoketest-%j.out
#SBATCH --error=/home/morg/students/sagiahrac/repos/olmo-rocm/slurm_output/olmoe-tiny-deepseek-smoketest-%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-210
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:amd:1
#SBATCH --time=00:15:00

export ROCM_HOME=/opt/rocm-6.4.3
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

cd /home/morg/students/sagiahrac/repos/olmo-rocm

source .venv/bin/activate

torchrun \
    --standalone \
    --nproc_per_node=1 \
    --master_port=29505 \
    scripts/train.py \
    configs/olmoe/ablations/olmoe-tiny-deepseek-smoketest.yaml
