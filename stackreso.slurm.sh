#!/bin/bash
#SBATCH --job-name=stack-reservoir
#SBATCH --partition=CPUQ
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=slurm.log
#SBATCH --account=share-ie-idi

set -e

module load GCC/8.3.0 CUDA/10.1.243
module load Python/3.7.4

set -x

env

# SLURM will set CUDA_VISIBLE_DEVICES for us which automatically selects the allocated GPU
# OpenCL will always see the allocated GPU as device 0
#flatspin-run -r worker -o {basepath} --worker-id ${{SLURM_ARRAY_TASK_ID}} --num-workers $((SLURM_ARRAY_TASK_MAX+1))
python /lustre1/home/lykkebo/stack_reservoir/run_sort_mnist.py
