#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --job-name=scene-job-capts
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=job-scripts/outs/slurm_output_capts.out
#SBATCH --error=job-scripts/outs/slurm_err_capts.err
#SBATCH --mail-type=END

source scenes/bin/activate

cd src/caption_generation
python llava-ov.py
python molmo.py
python phi-4.py
python pixtral.py
python qwen2.5-vl.py
