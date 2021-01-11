#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J SolarInspect
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err 
module load python3/3.8.2 
module load cuda/10.1 
module load cudnn/v8.0.4.30-prod-cuda-10.1
echo "Running script..."
python3 train.py