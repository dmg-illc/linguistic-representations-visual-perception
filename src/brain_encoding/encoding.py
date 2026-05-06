import os
from src.paths import ROOT
from src.utils import *
from src.fmri_responses.fmri_loading import ParticipantResponses
from src.indexing_and_formatting.image_indexing_utils import shared_subset
from src.indexing_and_formatting.model_names_and_paths import names_to_paths, encoders_to_embeddings
from src.brain_encoding.encoding_utils import pca, standardise_features
from src.brain_encoding.ridge import RidgeCVEstimator, compute_correlations
from typing import Dict, List
import numpy as np
import torch
from sklearn.model_selection import KFold
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--encoder', choices=['qwen3', 'llama', 'bert', 'gpt2', 'vit', 'resnet', 'kalm']) 
args = parser.parse_args()


class BrainEncoding:
    def __init__(self,
                 model_name: str,
                 brain_responses_lh: Dict[int, np.ndarray],
                 brain_responses_rh: Dict[int, np.ndarray],
                 image_indices: List[int],
                 alphas: np.ndarray,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.model_name = model_name
        self.image_indices = image_indices
        # stack responses as (n_samples, n_voxels)
        self.brain_responses_lh = np.stack([brain_responses_lh[idx] for idx in image_indices])
        self.brain_responses_rh = np.stack([brain_responses_rh[idx] for idx in image_indices])

        self.brain_responses_lh = torch.from_numpy(self.brain_responses_lh).to(self.device, dtype=self.dtype)
        self.brain_responses_rh = torch.from_numpy(self.brain_responses_rh).to(self.device, dtype=self.dtype)

        self.kf_outer = KFold(n_splits=5, shuffle=True, random_state=98)
        self.kf_splits = [(train_idx, test_idx) for (train_idx, test_idx) in self.kf_outer.split(self.image_indices)]

        
        self.alphas = torch.tensor(alphas, dtype=self.dtype, device=self.device)
        self.n_layers = None

    def get_preprocessed_features(self, fold_id: int):
        """
        Loads embeddings, runs StandardScaler + PCA on GPU and returns processed train/test features per layer.
        Saves cached preprocessing similar to original code.
        """
        if len(self.image_indices) > 900:
            target_path = ROOT / 'results/cached_preprocessing' / f"{self.model_name}-fold-{fold_id+1}.pkl"
        else:
            target_path = ROOT / 'results/cached_preprocessing' / f"scenes-only-{self.model_name}-fold-{fold_id+1}.pkl"

        if os.path.exists(target_path):
            preprocessed_xs = open_pickle(target_path)
            self.n_layers = len(preprocessed_xs['train'])
            return preprocessed_xs['train'], preprocessed_xs['test']

        model_embeddings = open_pickle(names_to_paths[self.model_name])
        n_layers = len(model_embeddings[self.image_indices[0]])
        self.n_layers = n_layers
        train_ids, test_ids = self.kf_splits[fold_id]
        preprocessed_xs = {'train': {}, 'test': {}}

        for layer in range(n_layers):
            emb_mat = np.stack([model_embeddings[img_id][layer] for img_id in self.image_indices])  # (n_samples, n_features)

            train_embs = emb_mat[train_ids, :]
            test_embs = emb_mat[test_ids, :]
            # print("Shape train embs:", train_embs.shape)

            # move to torch on device
            x_train = torch.from_numpy(train_embs).to(self.device, dtype=self.dtype)
            x_test = torch.from_numpy(test_embs).to(self.device, dtype=self.dtype)

            x_train_scaled, x_test_scaled = standardise_features(x_train, x_test)
            # PCA on GPU: compute SVD of X (n_samples x n_features). For n_features large, do economy SVD on covariance.
            # We'll compute SVD on x_train_scaled (shape n_samples x n_features). Use compact trick when n_samples < n_features.

            train_proj, test_proj = pca(x_train_scaled, x_test_scaled)
            # move back to cpu numpy for caching (to avoid storing huge tensors on GPU across runs)
            preprocessed_xs['train'][layer] = train_proj.cpu().numpy()
            preprocessed_xs['test'][layer] = test_proj.cpu().numpy()

            # free GPU memory
            del x_train, x_test, x_train_scaled, x_test_scaled
            torch.cuda.empty_cache()

        save_pickle(preprocessed_xs, target_path)
        return preprocessed_xs['train'], preprocessed_xs['test']


    def do_linear_encoding(self):
        """
        Main entry: loops over outer folds and layers, fits ridge on GPU in chunks over voxels, and returns accuracy arrays.
        """
        
        for i, (train_index, test_index) in enumerate(self.kf_splits):
            preprocessed_x_train, preprocessed_x_test = self.get_preprocessed_features(fold_id=i)

            if i == 0:
                folds_accuracies_lh = np.empty((5, self.n_layers, self.brain_responses_lh.shape[1]), dtype=np.float16)
                folds_accuracies_rh = np.empty((5, self.n_layers, self.brain_responses_rh.shape[1]), dtype=np.float16)

            y_train_lh = self.brain_responses_lh[train_index] 
            y_test_lh = self.brain_responses_lh[test_index]

            y_train_rh = self.brain_responses_rh[train_index]
            y_test_rh = self.brain_responses_rh[test_index]

            for layer in range(self.n_layers):
                x_train = torch.from_numpy(preprocessed_x_train[layer]).to(self.device, dtype=self.dtype) 
                x_test = torch.from_numpy(preprocessed_x_test[layer]).to(self.device, dtype=self.dtype)
       

                # finding best regularisation parameters through cv
                cv_lh = RidgeCVEstimator(self.alphas)
                cv_rh = RidgeCVEstimator(self.alphas)

                cv_lh.fit(X=x_train, Y=y_train_lh)
                cv_rh.fit(X=x_train, Y=y_train_rh)

                preds_lh = cv_lh.predict(X=x_test)
                preds_rh = cv_rh.predict(X=x_test)

                acc_lh = compute_correlations(preds_lh, y_test_lh)
                acc_rh = compute_correlations(preds_rh, y_test_rh)

                
                folds_accuracies_lh[i, layer, :] = acc_lh.cpu().numpy()  
                folds_accuracies_rh[i, layer, :] = acc_rh.cpu().numpy()

                del x_test, x_train, preds_lh, preds_rh, acc_lh, acc_rh, cv_lh, cv_rh
                    
            # print(f"End of fold {i+1}")

            
        return folds_accuracies_rh, folds_accuracies_lh
                    
def save_encoding_results(rh_accuracies, lh_accuracies, roi, participant: int, model_id: str, results_path: str):
    full_path = ROOT / results_path / f'{model_id}.pkl'
    if os.path.exists(full_path):
        results = open_pickle(full_path)
    else:
        results = {}

    current_dict = results

    for k in [str(participant), roi]:
        if k in current_dict:
            current_dict = current_dict[k]
        else:
            current_dict[k] = {}
            current_dict = current_dict[k]

    results[str(participant)][roi]['lh'] = lh_accuracies
    results[str(participant)][roi]['rh'] = rh_accuracies

    save_pickle(results, ROOT / results_path / f'{model_id}.pkl')


if __name__ == "__main__":

    RESULTS_PATH = "results/encoding/encoding_by_model"
    
    for participant in range(1,9):
        print(participant)

        part_resp = ParticipantResponses(participant=participant, noise_ceilings=False)

        roi = 'all'
        brain_resp_lh = part_resp.get_roi_activations(roi, hemisphere='lh')
        brain_resp_rh = part_resp.get_roi_activations(roi, hemisphere='rh')

        for model_name in encoders_to_embeddings[args.encoder]:
            print(model_name)
            enc_args = {'model_name': model_name, 
                    'brain_responses_lh': brain_resp_lh, 
                    'brain_responses_rh': brain_resp_rh, 
                    'image_indices': shared_subset,
                    'alphas': np.logspace(4, 25, 20)
                    }
            enc_pipeline = BrainEncoding(**enc_args)
            rh_accuracies, lh_accuracies = enc_pipeline.do_linear_encoding()

            save_encoding_results(rh_accuracies, lh_accuracies, roi, participant, model_name, RESULTS_PATH)
