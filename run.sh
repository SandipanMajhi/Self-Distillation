#!/bin/bash
#SBATCH --job-name=job_name # Job name
#SBATCH --output=/raid/<username>/output_%j.txt # Standard output
#SBATCH --error=/raid/<username>/error_%j.txt # Standard error
#SBATCH --partition=dgx2 # Partition name (modify if using a different node)
#SBATCH --qos=dgx2 # QoS (must match the partition)
#SBATCH --nodes=1 # Use 1 node
#SBATCH --ntasks-per-node=2 # Number of tasks per node (max 8)
#SBATCH --gres=gpu:2 # Number of GPUs (max 8)
#SBATCH --time=24:00:00 # Max runtime 72hrs(HH:MM:SS)

# Navigate to the working directory
cd /raid/<username>/
# Load necessary modules
source /scratch/apps/modules/init/bash
module load cuda/12.4
# module load python/3.10
# Activate your Python environment
source /home/<username>/miniconda3/bin/activate ultralytics-env
# Execute the training script
python train_model.py