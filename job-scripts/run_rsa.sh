#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=scene-job-embs
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=job-scripts/outs/slurm_output_encod.out
#SBATCH --error=job-scripts/outs/slurm_err_encod.err
#SBATCH --mail-type=END

source scenes/bin/activate


# Brain RSA
python src/rsa/brain_rsa.py -e bert
python src/rsa/brain_rsa.py -e llama
python src/rsa/brain_rsa.py -e gpt2
python src/rsa/brain_rsa.py -e kalm
python src/rsa/brain_rsa.py -e vit
python src/rsa/brain_rsa.py -e resnet

# Behavioural RSA
python src/rsa/simj_rsa.py -e bert
python src/rsa/simj_rsa.py -e llama
python src/rsa/simj_rsa.py -e gpt2
python src/rsa/simj_rsa.py -e kalm
python src/rsa/simj_rsa.py -e vit
python src/rsa/simj_rsa.py -e resnet