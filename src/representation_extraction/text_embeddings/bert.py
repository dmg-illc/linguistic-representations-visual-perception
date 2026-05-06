from transformers import BertTokenizer, BertModel
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
model_id = "google-bert/bert-large-uncased"
batch_size = 30

formatted_sentences = format_input_sentences(caption_type=args.captions)
image_ids = list(formatted_sentences.keys())
sentences = list(formatted_sentences.values())

tokenizer = BertTokenizer.from_pretrained(model_id, pad_token = "[PAD]")
# tokenizer.pad_token = tokenizer.eos_token
model = BertModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    
)

# representations from CLS token and averaged across tokes will be saved here
cls_token_reps = []
mean_token_reps = []

for batch in batchify(sentences, batch_size):
    inputs = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
    attention_mask = inputs['attention_mask'].cpu().numpy().astype(bool)
    output = model(
                    **inputs,
                    output_hidden_states=True     
                    )
    hidden_states = [h.detach().cpu().float().numpy() for h in output.hidden_states[1:]] # discarding embedding layer

    batch_size = hidden_states[0].shape[0]

    # looping over sentences
    for i in range(batch_size):
        mask = attention_mask[i]
        
        # Collect per-layer embeddings
        cls_per_layer = []
        mean_per_layer = []
        
        # looping over layers
        for layer_hidden in hidden_states:
            hs = layer_hidden[i][mask] 
            
            cls_per_layer.append(layer_hidden[i][0])             
            mean_per_layer.append(hs.mean(axis=0))    


        cls_token_reps.append(np.stack(cls_per_layer)) 
        mean_token_reps.append(np.stack(mean_per_layer))  


cls_tok_dict = {image_id: embedding for image_id, embedding in zip(image_ids, cls_token_reps)}
mean_tok_dict = {image_id: embedding for image_id, embedding in zip(image_ids, mean_token_reps)}

(ROOT / f'results/caption_embeddings/bert').mkdir(parents=True, exist_ok=True)

save_pickle(cls_tok_dict, ROOT / f'results/caption_embeddings/bert/bert_{args.captions}_cls.pkl')
save_pickle(mean_tok_dict, ROOT / f'results/caption_embeddings/bert/bert_{args.captions}_mean.pkl')

