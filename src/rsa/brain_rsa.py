from src.utils import *
from src.fmri_responses.fmri_loading_all_rois import ParticipantResponses
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr
from src.indexing_and_formatting.model_names_and_paths import names_to_paths
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.brain_encoding.intrinsic_dimensionality import IDAnalysis
from src.paths import ROOT
from src.rsa.rsa_utils import spearman_rowwise, compute_permutation_indices
import argparse
from src.indexing_and_formatting.model_names_and_paths import names_to_paths, encoders_to_embeddings
from src.indexing_and_formatting.image_indexing_utils import shared_subset



parser = argparse.ArgumentParser()
parser.add_argument('-e', '--encoder', choices=['qwen3', 'llama', 'bert', 'gpt2', 'vit', 'resnet', 'kalm']) 
args = parser.parse_args()

class BrainRSA:

    def __init__(self, model_id: str, image_indices: list, cache_rdm_path: str, rsa_results_path: str):
        
        self.image_indices = image_indices
        self.model_id = model_id
        n_images = len(self.image_indices)
        self.triu_indices = np.triu_indices(n=n_images, m=n_images, k=1)
        self.model_embeddings = open_pickle(names_to_paths[model_id])
        self.rois = ['all', 'faces', 'places', 'bodies']
        self.n_layers = len(self.model_embeddings[image_indices[0]])
        self.cached_rdm_path = str(cache_rdm_path)
        self.results_path = str(rsa_results_path)

    def precompute_brain_rdms(self):

        for participant in range(1, 9):

            # check if we already have precomputed rdms
            target_path = self.cached_rdm_path.format(participant = participant)
            if not os.path.exists(target_path):
                print(f"Precomputing brain RDMs for participant {participant}...")
                part_resp = ParticipantResponses(participant=participant, noise_ceilings=False)
                part_rdms = {}
                
                for roi in self.rois:
                    brain_resp_lh = part_resp.get_roi_activations(roi, hemisphere='lh')
                    brain_resp_rh = part_resp.get_roi_activations(roi, hemisphere='rh')
                    brain_mat_lh = np.stack([brain_resp_lh[image_id] for image_id in self.image_indices])
                    brain_mat_rh = np.stack([brain_resp_rh[image_id] for image_id in self.image_indices])
                    brain_resps = np.concatenate((brain_mat_lh, brain_mat_rh), axis=1)
                    brain_dist_mat = cosine_distances(brain_resps)
                    brain_rdm = brain_dist_mat[self.triu_indices].flatten()
                    part_rdms[roi] = brain_rdm

                    del brain_resp_lh, brain_resp_rh, brain_mat_lh, brain_mat_rh
                
                del part_resp
                save_pickle(part_rdms, target_path)

    def do_rsa(self):

        self.precompute_brain_rdms()

        results = {'rs': {roi: np.empty((self.n_layers, 8)) for roi in self.rois}, 
                   'pval': {roi: np.empty((self.n_layers, 8)) for roi in self.rois}}

        for layer in range(self.n_layers):
            print(f"Layer {layer+1}")
            emb_mat = np.stack([self.model_embeddings[img_id][layer] for img_id in self.image_indices])
            emb_dist_mat = cosine_distances(emb_mat)
            emb_rdm = emb_dist_mat[self.triu_indices].flatten()

            for participant in range(1, 9):
                target_path = self.cached_rdm_path.format(participant = participant)

                part_rdms = open_pickle(target_path)
                
                for roi in self.rois:
                    rs, pval = spearmanr(emb_rdm, part_rdms[roi])
                    results['rs'][roi][layer, (participant-1)] = rs
                    results['pval'][roi][layer, (participant-1)] = pval

                del part_rdms

        save_pickle(results, self.results_path.format(model_id = self.model_id))

    
    
    def do_permutation_test(self, permutation_indices, n_permutations=1000):

        self.precompute_brain_rdms()
        res = open_pickle(self.results_path.format(model_id = self.model_id))

        perm_results = {'all': {'pvals': np.empty(8), 'random_corr': np.empty(8), 'random_corr_std': np.empty(8)},
                        'faces': {'pvals': np.empty(8), 'random_corr': np.empty(8), 'random_corr_std': np.empty(8)}, 
                        'bodies': {'pvals': np.empty(8), 'random_corr': np.empty(8), 'random_corr_std': np.empty(8)}, 
                        'places': {'pvals': np.empty(8), 'random_corr': np.empty(8), 'random_corr_std': np.empty(8)}}
        
        for participant in range(1, 9):
                target_path = self.cached_rdm_path.format(participant = participant)
                part_rdms = open_pickle(target_path)

                for roi in self.rois:
                    best_layer = res['rs'][roi][:, participant-1].argmax()
                    emb_mat = np.stack([self.model_embeddings[img_id][best_layer] for img_id in self.image_indices])
                    emb_dist_mat = cosine_distances(emb_mat)
                    emb_rdm = emb_dist_mat[self.triu_indices].flatten()
                    permuted_rdms = np.stack([emb_rdm[permutation_indices[i]] for i in range(n_permutations)])
                    correlations = spearman_rowwise(permuted_rdms, part_rdms[roi])
                    observed_rs = res['rs'][roi][:, participant-1][best_layer]
                    all_correlations = np.append(correlations, observed_rs)
                    all_correlations.sort()

                    more_positive_corrs = (all_correlations >= observed_rs).sum()
                    more_negative_corrs = (all_correlations <= observed_rs).sum()

                    pval = min(more_positive_corrs, more_negative_corrs) / (n_permutations + 1)
                    perm_results[roi]['pvals'][participant-1] = pval
                    perm_results[roi]['random_corr'][participant-1] = correlations.mean()
                    perm_results[roi]['random_corr_std'][participant-1] = correlations.std()
                    
        save_pickle(perm_results, ROOT / f'results/brain_rsa/pvals/pvals_{self.model_id}.pkl')






class BrainRSAPlotsManager:

    def __init__(self, results_path: str, noise_ceilings: dict):
        self.results_path = str(results_path)
        self.brain_rois = ['all', 'faces', 'places', 'bodies']
        self.caption_types = {'coco': 'MS COCO', 'llava': 'LLaVA-OV', 'phi': 'Phi-4', 
                              'pixtral': 'Pixtral', 'qwen': 'Qwen2.5-VL', 'molmo': 'Molmo'}
        self.hatches = {'coco': '', 'llava': '..', 'phi': '//', 
                              'pixtral': '++', 'qwen': '\\\\//', 'molmo': '//..'}
        
        self.caption_embedders = {'qwen3': {'template': 'qwen3-{cap_type}-last', 'name': 'Qwen3 Embedding'},
                                'kalm': {'template': 'kalm-{cap_type}', 'name': 'KaLM Embedding'},
                                'bert': {'template': 'bert-{cap_type}-cls', 'name': 'BERT'},
                                'llama': {'template': 'llama-{cap_type}-last', 'name': 'Llama3'},
                                'gpt2': {'template': 'gpt2-{cap_type}-last', 'name': 'GPT-2'}}
                        
        self.noise_ceilings = noise_ceilings
        self.visual_features = {'vit': 'ViT', 'resnet': 'ResNet50'}
        self.training_types = {'-dino': 'DINO', '': 'ImageNet', '-clip': 'CLIP'}
    
    def layers_linechart(self, target_roi: str, embedder: str, caption_type: str, save=False, file_name = ''):

        """
        Plots a linechart specific for the given caption embedder and caption type.
        Each line in the plot corresponds to a specific participant.
        """

        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")
        
        if embedder not in self.caption_embedders:
            raise ValueError("Enter a valid caption embedder!")
        
        if caption_type not in self.caption_types:
            raise ValueError("Enter a valid caption type!")
        
        model_template = self.caption_embedders[embedder]['template'].format(cap_type = caption_type)
        model_name = self.caption_embedders[embedder]['name']

        results = open_pickle(self.results_path.format(model_id = model_template))
        
        plt.figure(figsize=(4.5,3.5), tight_layout=True)

        for participant in range(1,9):
            y = results['rs'][target_roi][:, (participant-1)]
            x = np.arange(len(y))
            plt.plot(x, y, alpha = 0.7, label = f"P{participant}")

        plt.plot(x, results['rs'][target_roi].mean(axis=1), lw=3, color='forestgreen', label='Avg.')
        plt.legend(ncols=3, loc='upper left')
        plt.ylim(-0.05, 0.4)
        plt.title(f"{model_name} {self.caption_types[caption_type]}")
        plt.grid(visible=True, axis='y')

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")
        
        plt.show()

    def avg_layer_linechart(self, embedder: str, target_roi: str, save=False, file_name = ''):
        """
        Plots a linechart specific for the given caption embedder where each
        line corresponds to the results for one caption type, averaged 
        across participants.
        """

        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")
        
        if embedder not in self.caption_embedders:
            raise ValueError("Enter a valid caption embedder!")

        plt.figure(figsize=(4.5,3.5), tight_layout=True)
        cmap = plt.get_cmap('tab10')
        model_template = self.caption_embedders[embedder]['template']
        for i, caption_type in enumerate(self.caption_types.keys()):
            
            results = open_pickle(self.results_path.format(model_id = model_template.format(cap_type = caption_type)))
            y = results['rs'][target_roi].mean(axis=1)
            x = np.arange(len(y))
            plt.plot(x, y, color = cmap(i), label = caption_type.capitalize())
            del results
        plt.legend(ncols=3, loc='upper left')
        plt.ylim(-0.1, 0.33)
        plt.title(self.caption_embedders[embedder]['name'])
        plt.grid(visible=True, axis='y')

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")

        plt.show()

    def linechart_intrinsic_dimensionality(self, embedder: str, target_roi: str, axspan, save=False, file_name = ''):

        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")
        
        if embedder not in self.caption_embedders:
            raise ValueError("Enter a valid caption embedder!")
        
        all_cap_res = []
        model_template = self.caption_embedders[embedder]['template']

        for i, caption_type in enumerate(self.caption_types.keys()):
            
            results = open_pickle(self.results_path.format(model_id = model_template.format(cap_type = caption_type)))
            cap_res = results['rs'][target_roi].mean(axis=1)
            all_cap_res.append(cap_res)
            del results

        rsa_avg = np.stack(all_cap_res).mean(axis=0)
        id_analysis = IDAnalysis(embedder)
        id_results = id_analysis.load_grade_results()
        n_layers, n_scales = id_results.shape
        x = np.arange(n_layers)
        y = id_results[x, id_analysis.best_scale_indices[id_analysis.model_name]]

        fig, ax1 = plt.subplots(figsize=(5, 3.5), tight_layout=True)

        color = 'tab:red'
        ax1.set_xlabel('# Layers')
        ax1.grid(visible=True, axis='y')
        ax1.set_ylabel(r'Brain Alignment ($\rho$)', color=color)
        ax1.plot(x, rsa_avg, '.-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Intrinsic dimensionality', color=color)  
        ax2.plot(x, y, '.-',color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        if axspan != None:
            ax2.axvspan(axspan[0], axspan[1], color="tab:red", alpha=0.1)

        title = f"{self.caption_embedders[embedder]['name']} — ROI: {target_roi.capitalize()}"
        plt.title(title)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")

        plt.show()


    def linechart_intrinsic_dimensionality_all_rois(self, embedder: str, axspan=None, save=False, file_name = ''):

        
        if embedder not in self.caption_embedders:
            raise ValueError("Enter a valid caption embedder!")
        
        all_cap_res = {'faces': [], 'bodies': [], 'places': []}
        model_template = self.caption_embedders[embedder]['template']

        for i, caption_type in enumerate(self.caption_types.keys()):

            for target_roi in ['faces', 'bodies', 'places']:
            
                results = open_pickle(self.results_path.format(model_id = model_template.format(cap_type = caption_type)))
                cap_res = results['rs'][target_roi].mean(axis=1)
                all_cap_res[target_roi].append(cap_res)
                del results

        rsa_avg = {k: np.stack(all_cap_res[k]).mean(axis=0) for k in all_cap_res}
        id_analysis = IDAnalysis(embedder)
        id_results = id_analysis.load_grade_results()
        n_layers, n_scales = id_results.shape
        x = np.arange(n_layers)
        y = id_results[x, id_analysis.best_scale_indices[id_analysis.model_name]]

        fig, ax1 = plt.subplots(figsize=(6, 3.5), tight_layout=True)
 
        ax1.set_xlabel('# Layers')
        ax1.grid(visible=True, axis='y')
        ax1.set_ylabel(r'Brain Alignment ($\rho$)')
        markers = ['-', '', '']
        colours = ['tab:red', 'tab:red', 'tab:green']
        for roi, col, marker in zip(rsa_avg, colours, markers):
            norm_factor = (self.noise_ceilings[roi]['ub'] + self.noise_ceilings[target_roi]['lb']) / 2
            ax1.plot(x, rsa_avg[roi]/norm_factor, f'{marker}-', markersize=3, lw=1.2, color=col, label=roi.capitalize(), alpha=0.8)
        ax1.tick_params(axis='y')
        # ax1.legend(ncol=1)
        ax1.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Intrinsic dimensionality', color=color)  
        ax2.plot(x, y, '.-',color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        title = f"{self.caption_embedders[embedder]['name']}"
        plt.title(title)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")

        plt.show()


    

    def barplot_all_encoders(self, target_roi: str, normalise=False, save=False, file_name = ''):

        """
        Big barplot grouped by caption encoder. For each, we show the average, best-layer
        RSA across participants, with error bars showing how much variance there is
        across participants. Noise ceilings are plotted as dashed lines. 
        """
        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")

        plt.figure(figsize=(6,3), tight_layout=True)
        cmap = plt.get_cmap('tab10')
        spacing_parameter = 0.14
        
        for i, emb in enumerate(self.caption_embedders):
            model_template = self.caption_embedders[emb]['template']
            for j, ct in enumerate(self.caption_types):
                results = open_pickle(self.results_path.format(model_id = model_template.format(cap_type=ct)))

                y = results['rs'][target_roi].max(axis=0)
                if normalise:
                    norm_factor = (self.noise_ceilings[target_roi]['ub'] + self.noise_ceilings[target_roi]['lb']) / 2
                    y /= norm_factor
                plt.bar(i+spacing_parameter*j, y.mean(), yerr = y.std(), color = cmap(j), alpha=0.8, width=spacing_parameter, hatch=self.hatches[ct], edgecolor='white')
                
                del results
        i +=1

        if not normalise:
            plt.hlines(self.noise_ceilings[target_roi]['ub'], -0.1, i-0.3, linestyles='dashed', color='black')
            plt.hlines(self.noise_ceilings[target_roi]['lb'], -0.1, i-0.3, linestyles='dashed', color='black')
            plt.ylim(-0.05, 0.67)
        else: 
            plt.ylim(0, 0.65)
        plt.ylabel(r"Spearman's $\rho$")
        plt.xticks(np.arange(i)+3*spacing_parameter, [self.caption_embedders[emb]['name'].replace(' ', '\n') for emb in self.caption_embedders])
        plt.title(f"Caption Embeddings — ROI: {target_roi.capitalize()}")
        plt.grid(visible=True, axis='y')
       
        legend_elements = [Patch(facecolor=cmap(i),
                            label=self.caption_types[ct], hatch=self.hatches[ct], edgecolor='white') for i, ct in enumerate(self.caption_types)]       
        
        plt.legend(handles=legend_elements, handleheight=1.2, ncols=3)
        plt.gca().set_axisbelow(True)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")

        plt.show()


    def barplot_visual_best_layer(self, target_roi: str, normalise=False, save=False, file_name = ''):

        """
        Barplot specific for a all visual models, showing an average of the best-layer
        RSA results across participants. Participant-specific values are plotted 
        as dots on top of the bars. Noise ceilings are displayed as dashed lines. 
        """

        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")

        plt.figure(figsize=(3,3), tight_layout=True)
        cmap = plt.get_cmap('tab10')
        spacing_parameter = 0.25
        hatches = ['..', '//', '']

        for i, mod in enumerate(self.visual_features):
            for j, tt in enumerate(self.training_types):
                results = open_pickle(self.results_path.format(model_id = f"{mod}{tt}"))
                y = results['rs'][target_roi].max(axis=0)
                if normalise:
                    norm_factor = (self.noise_ceilings[target_roi]['ub'] + self.noise_ceilings[target_roi]['lb']) / 2
                    y /= norm_factor
                plt.bar(i+spacing_parameter*j, y.mean(),yerr=y.std(), color = cmap(j+6), alpha=0.8, width=spacing_parameter, hatch = hatches[j], edgecolor='white')
                plt.text(i+spacing_parameter*j, y.mean()+y.std()+0.02, s='*', ha='center')
                del results
             
        if not normalise:
            plt.hlines(self.noise_ceilings[target_roi]['ub'], 0-0.1, i+0.35, linestyles='dashed', color='black')
            plt.hlines(self.noise_ceilings[target_roi]['lb'], 0-0.1, i+0.35, linestyles='dashed', color='black')
            plt.ylim(-0.05, 0.68)
        else:
            plt.ylim(0, 0.65)
        plt.ylabel(r"Spearman's $\rho$")
        plt.xticks(np.arange(len(self.visual_features))+spacing_parameter, [self.visual_features[vf]+'\n' for vf in self.visual_features])
        plt.title(f"Visual Features — ROI: {target_roi.capitalize()}")
        plt.grid(visible=True, axis='y')
        legend_elements = [Patch(facecolor=cmap(i+6),
                            label=self.training_types[tt], hatch = hatches[i], edgecolor='white') for i,tt  in enumerate(self.training_types)]
        plt.legend(handles=legend_elements, handleheight=1.2, ncols=2)
        plt.gca().set_axisbelow(True)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")

        plt.show()



if __name__ == "__main__":
        
        n_permutations = 1000
        permutation_indices = compute_permutation_indices(n_images=len(shared_subset), n_permutations=1000)
        
        for model_name in encoders_to_embeddings[args.encoder]:
            rsa = BrainRSA(model_name, shared_subset, ROOT / 'results/cached_brain_rdms' / "part-{participant}.pkl",
                ROOT / 'results/brain_rsa/{model_id}.pkl')
            rsa.do_permutation_test(permutation_indices, n_permutations=n_permutations)





