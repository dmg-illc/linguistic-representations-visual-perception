from transformers import CLIPVisionModel, AutoProcessor
import pickle
import argparse
from src.paths import ROOT
from PIL import Image
from src.utils import *
from src.format_sentences import batchify
import torch
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['vit']) 
args = parser.parse_args()

if args.model == 'vit':
    model_id = "openai/clip-vit-large-patch14-336"


batch_size = 10

img_file_names = os.listdir(ROOT / 'data/images/stimuli')
image_ids = [img_id.split('nsd')[-1].split('.png')[0].lstrip('0') for img_id in img_file_names]

model = CLIPVisionModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# representations from last token and averaged across tokes will be saved here

mean_feature_reps = []
cls_feature_reps = []

for batch in batchify(img_file_names, batch_size):
    images = [Image.open(ROOT / 'data/images/stimuli'/ name).convert('RGB')  for name in batch]
    inputs = processor(images = images, return_tensors="pt").to(model.device)
    output = model(
                    **inputs,
                    output_hidden_states=True     
                    )
    hidden_states = [h.detach().cpu().float().numpy() for h in output.hidden_states]

    batch_size = hidden_states[0].shape[0]

    # looping over images
    for i in range(batch_size):
        
        # Collect per-layer embeddings
        mean_per_layer = [layer_hidden[i].mean(axis=0) for layer_hidden in hidden_states]
        mean_feature_reps.append(np.stack(mean_per_layer))  # (num_layers * hidden_dim,)

        # cls
        cls_per_layer = [layer_hidden[i][0] for layer_hidden in hidden_states]
        cls_feature_reps.append(np.stack(cls_per_layer))  # (num_layers * hidden_dim,)

    del images, inputs, output, hidden_states



mean_ftr_dict = {image_id: features for image_id, features in zip(image_ids, mean_feature_reps)}
cls_dict = {image_id: features for image_id, features in zip(image_ids, cls_feature_reps)}
spatial_ftr_dict = {image_id: features for image_id, features in zip(image_ids, spatial_tokens_reps)}


(ROOT / f'results/image_features/clip').mkdir(parents=True, exist_ok=True)

save_pickle(cls_dict, ROOT / f'results/image_features/clip/clip_{args.model}_cls.pkl')


