import torch
from torch import Tensor
import argparse
from src.paths import ROOT
from src.utils import *
from src.indexing_and_formatting.format_sentences import format_input_sentences, batchify
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--captions', choices=['coco', 'llava-ov', 'molmo', 'phi-4', 'pixtral', 'qwen-vl']) 
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "Qwen/Qwen3-Embedding-8B"
batch_size = 5

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


formatted_sentences = format_input_sentences(caption_type=args.captions)
image_ids = list(formatted_sentences.keys())
sentences = list(formatted_sentences.values())

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModel.from_pretrained(model_id)

last_token_reps = []

for batch in batchify(sentences, batch_size):
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
    attention_mask = inputs['attention_mask']
    output = model(
                    **inputs,
                    output_hidden_states=True     
                    )
    hidden_states = output.hidden_states
    embeddings = [last_token_pool(state, attention_mask) for state in hidden_states[1:]] # the embedding layer is not informative, so we discard it


    batch_size = hidden_states[0].shape[0]

    # looping over sentences
    for i in range(batch_size):
        
        # Collect per-layer embeddings
        last_per_layer = []
        
        # looping over layers
        for layer_embeddings in embeddings:
            
            last_per_layer.append(layer_embeddings[i].detach().cpu().numpy())             # (hidden_dim,)

        last_token_reps.append(np.stack(last_per_layer))  # (num_layers * hidden_dim,)
    
    del inputs, hidden_states, embeddings



last_tok_dict = {image_id: embedding for image_id, embedding in zip(image_ids, last_token_reps)}

(ROOT / f'results/caption_embeddings/qwen3').mkdir(parents=True, exist_ok=True)

save_pickle(last_tok_dict, ROOT / f'results/caption_embeddings/qwen3/qwen3_{args.captions}_last.pkl')
