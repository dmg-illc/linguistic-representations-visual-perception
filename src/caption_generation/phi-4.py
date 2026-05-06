# only runs with transformers==4.48.2

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

# Define model path
model_id = "microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    # if you do not use Ampere or later GPUs, change attention to "eager"
    _attn_implementation='eager',
).cuda()

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_id)

captions_dict = {}

for img_file in image_files:
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    # Part 1: Image Processing
    # print("\n--- IMAGE PROCESSING ---")
   
    prompt = f'{user_prompt}<|image_1|>Provide a brief description of this image{prompt_suffix}{assistant_prompt}'
    # print(f'>>> Prompt\n{prompt}')

    # Download and open image
    image = Image.open(ROOT / img_dir / img_file)
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

    # Generate response
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        generation_config=generation_config,
        temperature = 0
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # print(f'>>> Response\n{response}')
    captions_dict[img_file.split('nsd')[-1].split('.png')[0].lstrip('0')] = response

    del prompt, image, inputs, generate_ids, response

(ROOT / 'results/generated_captions/phi').mkdir(parents=True, exist_ok=True)

save_pickle(captions_dict, ROOT / f'results/generated_captions/phi/phi-captions.pkl')
    

    