from src.utils import *
from src.paths import ROOT
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr
from src.indexing_and_formatting.image_indexing_utils import judj_subset
from src.indexing_and_formatting.model_names_and_paths import names_to_paths, encoders_to_embeddings
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.brain_encoding.intrinsic_dimensionality import IDAnalysis
from src.rsa.rsa_utils import spearman_rowwise, compute_permutation_indices

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--encoder', choices=['qwen3', 'llama', 'bert', 'gpt2', 'vit', 'resnet', 'kalm']) 
args = parser.parse_args()


class SimjRSA:

    def __init__(self, model_name: str, image_indices: list, rsa_results_path: str):

        self.image_indices = image_indices
        self.model_name = model_name
        n_images = len(self.image_indices)
        self.triu_indices = np.triu_indices(n=n_images, m=n_images, k=1)
        self.model_embeddings = open_pickle(names_to_paths[model_name])
        self.n_layers = len(self.model_embeddings[image_indices[0]])
        self.results_path = str(rsa_results_path)


    def load_simj_rdms(self):
        rdms = open_pickle(ROOT / 'results/judj_rsa/simj.pkl')
        return rdms

    def do_rsa(self):

        simj_rdms = self.load_simj_rdms()

        results = {'rs': np.empty((self.n_layers, 8)), 
                   'pval': np.empty((self.n_layers, 8))}

        for layer in range(self.n_layers):
            print(f"Layer {layer+1}")
            emb_mat = np.stack([self.model_embeddings[img_id][layer] for img_id in self.image_indices])
            emb_dist_mat = cosine_distances(emb_mat)
            emb_rdm = emb_dist_mat[self.triu_indices].flatten()

            for participant in range(1, 9):
                part_rdm = simj_rdms[str(participant)]

                rs, pval = spearmanr(emb_rdm, part_rdm)
                # print(rs)
                results['rs'][layer, (participant-1)] = rs
                results['pval'][layer, (participant-1)] = pval

        save_pickle(results, self.results_path.format(model_name=self.model_name))

    def do_permutation_test(self, permutation_indices, n_permutations=1000):
        simj_rdms = self.load_simj_rdms()
        res = open_pickle(self.results_path.format(model_name = self.model_name))

        perm_results = {'pvals': np.empty(8), 'random_corr': np.empty(8), 'random_corr_std': np.empty(8)}
        
        for participant in range(1, 9):
            best_layer = res['rs'][:, participant-1].argmax()
            emb_mat = np.stack([self.model_embeddings[img_id][best_layer] for img_id in self.image_indices])
            emb_dist_mat = cosine_distances(emb_mat)
            emb_rdm = emb_dist_mat[self.triu_indices].flatten()
            permuted_rdms = np.stack([emb_rdm[permutation_indices[i]] for i in range(n_permutations)])
            correlations = spearman_rowwise(permuted_rdms, simj_rdms[str(participant)])
            observed_rs = res['rs'][:, participant-1][best_layer]
            all_correlations = np.append(correlations, observed_rs)
            all_correlations.sort()

            more_positive_corrs = (all_correlations >= observed_rs).sum()
            more_negative_corrs = (all_correlations <= observed_rs).sum()

            pval = min(more_positive_corrs, more_negative_corrs) / (n_permutations + 1)
            perm_results['pvals'][participant-1] = pval
            perm_results['random_corr'][participant-1] = correlations.mean()
            perm_results['random_corr_std'][participant-1] = correlations.std()
            
        save_pickle(perm_results, ROOT / f'results/judj_rsa/pvals/pvals_{self.model_name}.pkl')




class SimjRSAPlotsManager:

    def __init__(self, noise_ceilings: dict):

        self.results_path = str(ROOT / 'results/judj_rsa/{model_id}.pkl')
        self.caption_types = {'coco': 'MS COCO', 'llava': 'LLaVA-OV', 'phi': 'Phi-4', 
                              'pixtral': 'Pixtral', 'qwen': 'Qwen2.5-VL', 'molmo': 'Molmo'}
                            #   'loc-narr': 'LocNarr', 'coco-avg': 'MS COCO avg.'

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

    def barplot_all_lms(self, save=False, file_name = ''):

        """
        Creates a big barplot containing all best-layer accuracies.
        It is organised by embedder and it plots an average across 
        participants, with the error bars indicating the standard
        deviation across participants.
        """

        plt.figure(figsize=(6,3), tight_layout=True)
        cmap = plt.get_cmap('tab10')
        spacing_parameter = 0.14
        
        for i, emb in enumerate(self.caption_embedders):
            model_template = self.caption_embedders[emb]['template']

            for j, ct in enumerate(self.caption_types):

                results = open_pickle(self.results_path.format(model_id=model_template.format(cap_type = ct)))
                y = results['rs'].max(axis=0)
                plt.bar(i+spacing_parameter*j, y.mean(), yerr = y.std(), color = cmap(j), alpha=0.8, width=spacing_parameter, hatch = self.hatches[ct], edgecolor='white')
                skip = {('bert','coco'), ('llama','coco'), ('gpt2','coco')}
                if (emb, ct) not in skip:
                    plt.text(i+spacing_parameter*j, y.mean()+y.std()+0.02, s='*', ha='center')

                del results
                
        plt.hlines(self.noise_ceilings['ub'], -0.1, len(self.caption_embedders)-0.3, linestyles='dashed', color='black')
        plt.hlines(self.noise_ceilings['lb'], -0.1, len(self.caption_embedders)-0.3, linestyles='dashed', color='black')
        plt.ylim(0, 0.58)
        embedders = [self.caption_embedders[emb]['name'].replace(' ', '\n') for emb in self.caption_embedders]
        plt.xticks(np.arange(len(self.caption_embedders))+3*spacing_parameter, embedders)
        plt.ylabel(r"Spearman's $\rho$")

        plt.title("Caption Embeddings")
        plt.grid(visible=True, axis='y')
        legend_elements = [Patch(facecolor=cmap(i),
                            label=self.caption_types[ct], hatch = self.hatches[ct], edgecolor='white') for i, ct in enumerate(self.caption_types)]
        plt.legend(handles=legend_elements, handleheight=1.2, ncols=3)
        plt.gca().set_axisbelow(True)
        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")


        plt.show()

    
    def barplot_all_vf(self, save=False, file_name = ''):
        """
        Creates a barplot showing alignment for visual features. 
        Each bar represents results for a specific model, averaged
        across participants. Error bars represent the standard 
        deviation across participants. 
        """

        plt.figure(figsize=(3,3), tight_layout=True)
        cmap = plt.get_cmap('tab10')
        spacing_parameter = 0.25
        hatches = ['..', '//', '']

        for i, mod in enumerate(self.visual_features):
            for j, tt in enumerate(self.training_types):

                results = open_pickle(ROOT / 'results/judj_rsa' / f'{mod+tt}.pkl')
                y = results['rs'].max(axis=0)
                # print(results['rs'])
                plt.bar(i+spacing_parameter*j, y.mean(), yerr = y.std(), color = cmap(j+6), alpha=0.8, width=spacing_parameter, hatch = hatches[j], edgecolor='white')
                plt.text(i+spacing_parameter*j, y.mean()+y.std()+0.02, s='*', ha='center')

                del results
                
        plt.hlines(self.noise_ceilings['ub'], -0.06, len(self.visual_features)-0.65, linestyles='dashed', color='black')
        plt.hlines(self.noise_ceilings['lb'], -0.06, len(self.visual_features)-0.65, linestyles='dashed', color='black')
        plt.ylim(0, 0.58)
        plt.ylabel(r"Spearman's $\rho$")
        plt.xticks(np.arange(len(self.visual_features))+spacing_parameter,[self.visual_features[vf]+'\n' for vf in self.visual_features])
        plt.title(f"Visual Features")
        plt.grid(visible=True, axis='y')
        legend_elements = [Patch(facecolor=cmap(i+6),
                            label=self.training_types[tt], hatch = hatches[i], edgecolor='white') for i, tt in enumerate(self.training_types)]
        plt.legend(handles=legend_elements, handleheight=1.2, ncols=1)
        plt.gca().set_axisbelow(True)
        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")


        plt.show()


    def plot_avg_layers(self, embedder: str, save=False, file_name = ''):
        """
        Creates a linechart where every line represents the
        alignment progression through layers for one
        specific caption types. All results are shown for the
        given caption embedder. 
        """
        model_template = self.caption_embedders[embedder]['template']
        plt.figure(figsize=(4.5,3.5), tight_layout=True)
        cmap = plt.get_cmap('tab10')

        for i, caption_type in enumerate(self.caption_types):
            results = open_pickle(ROOT / 'results/judj_rsa' / f'{model_template.format(cap_type = caption_type)}.pkl')
            y = results['rs'].mean(axis=1)
            x = np.arange(len(y))
            plt.plot(x, y, color = cmap(i), label = caption_type.capitalize())
            del results
        plt.legend(ncols=3)
        plt.ylim(-0.03, 0.28)
        plt.title(self.caption_embedders[embedder]['name'])
        plt.grid(visible=True, axis='y')
        plt.show()

    def linechart_intrinsic_dimensionality(self, embedder: str, axspan=None, save=False, file_name = ''):

        if embedder not in self.caption_embedders:
            raise ValueError("Enter a valid caption embedder!")
        
        model_template = self.caption_embedders[embedder]['template']
        all_cap_res = []
        for i, caption_type in enumerate(self.caption_types):
            results = open_pickle(ROOT / 'results/judj_rsa' / f'{model_template.format(cap_type = caption_type)}.pkl')
            y = results['rs'].mean(axis=1)
            all_cap_res.append(y)
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
        ax1.set_ylabel(r'Behavioural Alignment ($\rho$)', color=color)
        ax1.plot(x, rsa_avg, '.-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Intrinsic dimensionality', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, y, '.-',color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        if axspan != None:
            ax2.axvspan(axspan[0], axspan[1], color="tab:red", alpha=0.1)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        title = f"{self.caption_embedders[embedder]['name']}"
        plt.title(title)
        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")

        plt.show()

if __name__ == "__main__":
        
        n_permutations = 1000
        permutation_indices = compute_permutation_indices(n_images=len(judj_subset), n_permutations=1000)
        
        for model_name in encoders_to_embeddings[args.encoder]:
            my_rsa = SimjRSA(model_name, judj_subset, rsa_results_path=ROOT / "results/judj_rsa/{model_name}.pkl")
            my_rsa.do_permutation_test(permutation_indices, n_permutations=n_permutations)
