from sentence_transformers import SentenceTransformer
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
model_id = "tencent/KaLM-Embedding-Gemma3-12B-2511"
batch_size = 1

formatted_sentences = format_input_sentences(caption_type=args.captions)
image_ids = list(formatted_sentences.keys())
sentences = list(formatted_sentences.values())

model = SentenceTransformer("tencent/KaLM-Embedding-Gemma3-12B-2511")
transformer, pooling, normalise = model[0].auto_model, model[1], model[2]
transformer.config.output_hidden_states = True

# representations from last token and averaged across tokes will be saved here
reps = []

with torch.no_grad():
    for batch in batchify(sentences, batch_size):
        print('Starting new batch')
        inputs = model.tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)

        # inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
        attention_mask = inputs['attention_mask']
        outputs = transformer(**inputs.to(model.device))
        norm_features = []
        for hidden_state in outputs.hidden_states[1:]:
            features = {'token_embeddings': hidden_state,
                        'attention_mask': attention_mask}
            pooled_features = pooling(features)
            del features
            normalised_features = normalise(pooled_features)
            norm_features.append(normalised_features['sentence_embedding'])
            del pooled_features

        current_bs = len(batch)

        for cap in range(current_bs):
            cap_reps = []
            for nf in norm_features:
                cap_reps.append(nf[cap].detach().cpu().numpy())
            reps.append(np.stack(cap_reps))

        del inputs, outputs, attention_mask, normalised_features, norm_features
        

emb_dict = {image_id: embedding for image_id, embedding in zip(image_ids, reps)}

(ROOT / f'results/caption_embeddings/kalm').mkdir(parents=True, exist_ok=True)

save_pickle(emb_dict, ROOT / f'results/caption_embeddings/kalm/kalm_{args.captions}.pkl')

