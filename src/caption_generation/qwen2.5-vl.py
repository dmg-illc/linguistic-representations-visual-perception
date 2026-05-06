from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
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

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

captions_dict = {}

for img_file in image_files:

    messages = [

        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(img_dir / img_file)},
                {"type": "text", "text": "Provide a short description of this image (max. 2 sentences). Do not include interpretations ('This suggests/This appears to be'). Do not include expressions like 'This is an image/close-up of'"}
            ],
        },
    ]

    # set use audio in video
    USE_AUDIO_IN_VIDEO = False

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128, temperature = 0.001)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    captions_dict[img_file.split('nsd')[-1].split('.png')[0].lstrip('0')] = output_text[0]
    del messages, image_inputs, video_inputs, generated_ids, generated_ids_trimmed, output_text
 
(ROOT / 'results/generated_captions/qwen-vl').mkdir(parents=True, exist_ok=True)

save_pickle(captions_dict, ROOT / f'results/generated_captions/qwen-vl/qwen-vl-captions.pkl')