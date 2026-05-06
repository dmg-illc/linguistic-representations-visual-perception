from huggingface_hub import snapshot_download
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from src.paths import ROOT
from PIL import  Image
import os
from random import sample, seed
from src.utils import *

img_dir = ROOT / "data/images/stimuli"
image_files = os.listdir(img_dir)
seed(3)
img_file_subset = sample(image_files, 20)

mistral_models_path = Path('/scratch-shared/abavaresco/mistral_models/Pixstral')
mistral_models_path.mkdir(parents=True, exist_ok=True)

# downloading weights (if not there)
snapshot_download(repo_id="mistralai/Pixtral-12B-2409", 
    allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"], 
    local_dir=mistral_models_path, token=["my_token"])

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
model = Transformer.from_folder(mistral_models_path)

captions_dict = {}

for img_file in image_files:
    prompt = "Provide a short (max. 2 sentences) description of the image. Start with the description directly, without 'The image shows' or 'In the image'."

    completion_request = ChatCompletionRequest(messages=[UserMessage(content=[ImageChunk(image=Image.open(ROOT / img_dir / img_file)), TextChunk(text=prompt)])])

    encoded = tokenizer.encode_chat_completion(completion_request)

    images = encoded.images
    tokens = encoded.tokens

    out_tokens, _ = generate([tokens], model, images=[images], max_tokens=256, temperature=0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    generated_text = tokenizer.decode(out_tokens[0])

    captions_dict[img_file.split('nsd')[-1].split('.png')[0].lstrip('0')] = generated_text

    del completion_request, encoded, images, tokens, out_tokens, generated_text

(ROOT / 'results/generated_captions/pixtral').mkdir(parents=True, exist_ok=True)

save_pickle(captions_dict, ROOT / f'results/generated_captions/pixtral/pixtral-captions.pkl')