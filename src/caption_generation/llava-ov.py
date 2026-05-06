from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
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

model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

processor = AutoProcessor.from_pretrained(model_id) 
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

captions_dict = {}

for img_file in image_files:

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(ROOT / img_dir / img_file)},
                {"type": "text", "text": "Describe the content of this image in a couple of sentences."},
            ],
        },
    ]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=200, temperature = 0)
    output_text = processor.decode(output[0], skip_special_tokens=True).split('assistant\n')[1]
    captions_dict[img_file.split('nsd')[-1].split('.png')[0].lstrip('0')] = output_text
    del conversation, inputs, output
 
(ROOT / 'results/generated_captions/llava').mkdir(parents=True, exist_ok=True)

save_pickle(captions_dict, ROOT / f'results/generated_captions/llava/llava-captions.pkl')