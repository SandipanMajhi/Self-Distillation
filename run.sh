#!/bin/bash
#SBATCH --job-name=distil_test01 # Job name
#SBATCH --output=/dgx001/plabanai/Sandipan_AI/Self-Distillation/Outputs/distil_test01.txt # Standard output
#SBATCH --error=/dgx001/plabanai/Sandipan_AI/Self-Distillation/Outputs/error01.txt # Standard error
#SBATCH --partition=dgx2 # Partition name (modify if using a different node)
#SBATCH --qos=dgx2 # QoS (must match the partition)
#SBATCH --nodes=1 # Use 1 node
#SBATCH --ntasks-per-node=1 # Number of tasks per node (max 8)
#SBATCH --gres=gpu:1 # Number of GPUs (max 8)
#SBATCH --time=24:00:00 # Max runtime 72hrs(HH:MM:SS)

# Navigate to the working directory
# cd /raid/<username>/
cd /dgx001/plabanai/Sandipan_AI/Self-Distillation
# Load necessary modules
# source /scratch/apps/modules/init/bash
# module load cuda/12.4
# module load python/3.10
# Activate your Python environment
# source /home/<username>/miniconda3/bin/activate ultralytics-env
# Execute the training script
# python train_model.py

source /dgx001/plabanai/Sandipan_AI/Self-Distillation/self_distil/bin/activate
python3 main.py