from src.paths import ROOT 
from src.utils import * 
from scipy.stats import false_discovery_control
from src.indexing_and_formatting.model_names_and_paths import encoders_to_embeddings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--encoder', choices=['qwen3', 'llama', 'bert', 'gpt2', 'kalm', 'vit', 'resnet']) 
args = parser.parse_args()


def compute_significance_masks(model_id):

    results = open_pickle(ROOT / f"results/encoding/encoding_by_model/{model_id}.pkl" )
    permutations = open_pickle(ROOT / f"results/encoding/permutations_by_model/{model_id}.pkl" )
    masks_path = ROOT / f"results/encoding/significance_masks/{model_id}.pkl"

    print("Computing masks...")
    
    masks = {}
    for part in range(1,9):
        masks[str(part)] = {}
        for hemisphere in ['lh', 'rh']:
            res = results[str(part)]['all'][hemisphere].mean(axis = 0).max(axis=0)
            perms = permutations[str(part)]['all'][hemisphere].mean(axis=0)
            n_permutations = perms.shape[0]
            p_values = (perms >= res).sum(axis = 0) / (n_permutations + 1)
            corrected_pvalues = false_discovery_control(p_values)
            mask = (corrected_pvalues<0.05)
            masks[str(part)][hemisphere] = mask

    del permutations, results

    print("Saving computed masks...")
    save_pickle(masks, masks_path)

if __name__ == "__main__":
    

    for model_name in encoders_to_embeddings[args.encoder]:
        print(model_name)
        compute_significance_masks(model_id=model_name)