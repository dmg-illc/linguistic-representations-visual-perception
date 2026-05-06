from src.paths import ROOT
from src.utils import *
import numpy as np

def remove_prefixes(text, prefixes):
    """
    Removes the first matching prefix from the beginning of the text.

    Parameters:
        text (str): The input string.
        prefixes (list of str): List of substrings to check as prefixes.

    Returns:
        str: The input string with the first matching prefix removed.
    """
    for prefix in prefixes:
        if text.startswith(prefix):
            new_text = text[len(prefix):]
            return new_text[0].upper() + new_text[1:]
    return text.lstrip(' ')[0].upper() + text.lstrip(' ')[1:]


def format_input_sentences(caption_type: str):

    '''
        Takes as input a string referring to the caption types and returns
        a dict where keys are NSD image ids and values are sentences. 
    '''

    

    if caption_type == 'coco':
        formatted_captions = {}
        all_captions = open_json(ROOT / "data/images/cleaned_coco_annotations/captions.json")
        
        np.random.seed(3)

        for image_id in all_captions:
            image_captions = all_captions[image_id]
            sampled_index = np.random.choice(5, 1)
            formatted_captions[image_id] = image_captions[sampled_index.item()]

   
    elif caption_type == 'llava':
        formatted_captions = open_pickle(ROOT / 'results/generated_captions/llava/llava-captions.pkl')

    elif caption_type == 'molmo':
        molmo_outputs = open_pickle(ROOT / 'results/generated_captions/molmo/molmo-captions.pkl')
        formatted_captions = {k: remove_prefixes(molmo_outputs[k].lstrip(' '), ['The image shows ', 'The image depicts ', 'The image displays ']) for k in molmo_outputs}

    elif caption_type == 'phi':
        formatted_captions = open_pickle(ROOT / 'results/generated_captions/phi/phi-captions.pkl')

    elif caption_type == 'pixtral':
        formatted_captions = open_pickle(ROOT / 'results/generated_captions/pixtral/pixtral-captions.pkl')

    elif caption_type == 'qwen':
        formatted_captions = open_pickle(ROOT / 'results/generated_captions/qwen-vl/qwen-vl-captions.pkl')

    
    return formatted_captions   
         

def batchify(data, batch_size):
    '''Yield successive batches of size `batch_size` from data.'''
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]



