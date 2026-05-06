from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import pickle
import argparse
from src.paths import ROOT
from src.utils import *
import torch
from PIL import Image
import numpy as np
import os
from random import sample, seed

img_dir = ROOT / "data/images/stimuli"
image_files = os.listdir(img_dir)
seed(3)
img_file_subset = sample(image_files, 20)

model_id = "allenai/Molmo-7B-D-0924"

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

captions_dict = {}

for img_file in image_files:

    inputs = processor.process(
    images = [Image.open(img_dir / img_file)],
    text="Provide a brief description (max. 2 sentences) of this image. The description should be strictly factual, without any interpretation of the visual elements.", return_tensors="pt",
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    captions_dict[img_file.split('nsd')[-1].split('.png')[0].lstrip('0')] = generated_text

    del inputs, output, generated_tokens

(ROOT / 'results/generated_captions/molmo').mkdir(parents=True, exist_ok=True)

save_pickle(captions_dict, ROOT / f'results/generated_captions/molmo/molmo-captions.pkl')