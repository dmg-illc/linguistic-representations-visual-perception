from src.paths import ROOT
from src.utils import *
from src.indexing_and_formatting.image_indexing_utils import shared_subset
from src.indexing_and_formatting.format_sentences import format_input_sentences
import math
from typing import List
import numpy as np
import os

import torch
import torch.nn.functional as F
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


def compute_batch_perplexities(
    texts: List[str],
    model_id: str,
    batch_size: int = 8,
    device: str = None,
) -> List[float]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer & model
    tokenizer = MistralCommonBackend.from_pretrained(model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
    )
    model.to(device)
    model.eval()

    # Ensure we have a pad token 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    all_perplexities: List[float] = []

    # Process texts in mini-batches
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,      
            truncation=True,   
        )

        input_ids = encodings["input_ids"].to(device)          
        attention_mask = encodings["attention_mask"].to(device) 
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  

        # Shift logits and labels so that token t predicts token t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Flatten for cross-entropy
        vocab_size = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        )  

        # Reshape 
        loss_tokens = loss_flat.view(shift_labels.size(0), shift_labels.size(1))

        # Mask out padding positions
        loss_tokens = loss_tokens * shift_mask.float()

        # Sum over tokens, divide by number of valid tokens -> mean NLL per token
        token_counts = shift_mask.sum(dim=1)  
        nll_per_sequence = loss_tokens.sum(dim=1) / token_counts  # (B,)

        # Perplexity = exp(mean NLL per token)
        batch_ppls = torch.exp(nll_per_sequence).tolist()
        all_perplexities.extend(batch_ppls)


    return np.array(all_perplexities)


if __name__ == "__main__":
    if not os.path.isdir(ROOT / "results/caption_metrics/perplexity"):
        os.mkdir(ROOT / "results/caption_metrics/perplexity")

    for caption_type in ['molmo', 'phi', 'llava', 'coco', 'pixtral', 'qwen']:

        captions = format_input_sentences(caption_type=caption_type)
        sentences = [captions[id] for id in shared_subset]

        ppls = compute_batch_perplexities(
            sentences,
            model_id="mistralai/Ministral-3-8B-Base-2512",
            batch_size=50,
        )

        save_pickle(ppls, ROOT / f"results/caption_metrics/perplexity/{caption_type}_perplexity.pkl")
    

