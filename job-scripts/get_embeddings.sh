#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=scene-job-embs
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=job-scripts/outs/slurm_output_embs.out
#SBATCH --error=job-scripts/outs/slurm_err_embs.err
#SBATCH --mail-type=END


source scenes/bin/activate

# Extract text embeddings
python src/representation_extraction/text_embeddings/bert.py -c coco
python src/representation_extraction/text_embeddings/bert.py -c llava-ov
python src/representation_extraction/text_embeddings/bert.py -c molmo
python src/representation_extraction/text_embeddings/bert.py -c pixtral
python src/representation_extraction/text_embeddings/bert.py -c phi-4
python src/representation_extraction/text_embeddings/bert.py -c qwen-vl

python src/representation_extraction/text_embeddings/gpt-2.py -c coco
python src/representation_extraction/text_embeddings/gpt-2.py -c llava-ov
python src/representation_extraction/text_embeddings/gpt-2.py -c molmo
python src/representation_extraction/text_embeddings/gpt-2.py -c pixtral
python src/representation_extraction/text_embeddings/gpt-2.py -c phi-4
python src/representation_extraction/text_embeddings/gpt-2.py -c qwen-vl

python src/representation_extraction/text_embeddings/kalm.py -c coco
python src/representation_extraction/text_embeddings/kalm.py -c llava-ov
python src/representation_extraction/text_embeddings/kalm.py -c molmo
python src/representation_extraction/text_embeddings/kalm.py -c pixtral
python src/representation_extraction/text_embeddings/kalm.py -c phi-4
python src/representation_extraction/text_embeddings/kalm.py -c qwen-vl

python src/representation_extraction/text_embeddings/llama.py -c coco
python src/representation_extraction/text_embeddings/llama.py -c llava-ov
python src/representation_extraction/text_embeddings/llama.py -c molmo
python src/representation_extraction/text_embeddings/llama.py -c pixtral
python src/representation_extraction/text_embeddings/llama.py -c phi-4
python src/representation_extraction/text_embeddings/llama.py -c qwen-vl

python src/representation_extraction/text_embeddings/qwen3.py -c coco
python src/representation_extraction/text_embeddings/qwen3.py -c llava-ov
python src/representation_extraction/text_embeddings/qwen3.py -c molmo
python src/representation_extraction/text_embeddings/qwen3.py -c pixtral
python src/representation_extraction/text_embeddings/qwen3.py -c phi-4
python src/representation_extraction/text_embeddings/qwen3.py -c qwen-vl

# Extracting visual features
python src/representation_extraction/image_features/resnet50_clip.py
python src/representation_extraction/image_features/resnet50_dino.py
python src/representation_extraction/image_features/resnet50.py
python src/representation_extraction/image_features/vit_clip.py
src/representation_extraction/image_features/vit_dino.py
src/representation_extraction/image_features/vit.py