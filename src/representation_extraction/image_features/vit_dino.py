import torch
from transformers import AutoImageProcessor, AutoModel
from src.paths import ROOT
from src.utils import *
from src.indexing_and_formatting.format_sentences import batchify
import numpy as np
from PIL import Image
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = 'facebook/dinov2-large'
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)

img_file_names = os.listdir(ROOT / 'data/images/stimuli')
image_ids = [img_id.split('nsd')[-1].split('.png')[0].lstrip('0') for img_id in img_file_names]

mean_feature_reps = []
batch_size = 20

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

    del images, inputs, output, hidden_states




mean_ftr_dict = {image_id: features for image_id, features in zip(image_ids, mean_feature_reps)}

(ROOT / f'results/image_features/dino').mkdir(parents=True, exist_ok=True)

save_pickle(mean_ftr_dict, ROOT / f'results/image_features/dino/dino.pkl')
