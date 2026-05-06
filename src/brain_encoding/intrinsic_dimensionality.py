from src.utils import *
from src.paths import ROOT
from dadapy.data import Data
from src.indexing_and_formatting.model_names_and_paths import names_to_paths, encoders_to_embeddings
from src.indexing_and_formatting.image_indexing_utils import shared_subset
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

class IDAnalysis():

    def __init__(self, model_name):

        if model_name not in encoders_to_embeddings:
            raise ValueError("Invalid model name!")
        
        self.model_name = model_name
        self.model_ids = encoders_to_embeddings[model_name]
        self.names_to_titles = {'bert': 'BERT', 'gpt2': 'GPT-2', 
                    'llama': 'Llama3', 'qwen3': 'Qwen3 Embeddings', 
                    'kalm': 'Kalm Embeddings'}
        self.best_scale_indices = {'bert': np.array([7, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7]),
        'gpt2': np.array([6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3]),
        'llama': np.array([5, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 4, 4, 6, 6, 6]),
        'qwen3': np.array([6, 6, 6, 6, 6, 6, 6, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 7, 6, 6, 6, 4, 4, 3, 3, 3, 5, 5, 5, 5, 6, 6, 6]),
        'kalm': np.array([6, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 6,6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])}


    def load_embeddings(self):

        """
            Loads embeddings for a certain model. Embeddings from all caption types are
            considered (COCO, LLaVA NeXT, Pixtral, Phi-4, Molmo, Qwen2.5-VL)
        """

        embeddings = []
        
        for model_id in self.model_ids:

            model_dict = open_pickle(names_to_paths[model_id])
            model_embs = np.stack([model_dict[img_id] for img_id in shared_subset])
            embeddings.append(model_embs)
            del model_dict, model_embs
        
        return np.concatenate(embeddings, axis=0)
    
    def compute_pca(self):

        embeddings = self.load_embeddings()
        n_samples, n_layers, initial_ndim = embeddings.shape
        dims = np.empty(n_layers)
        for layer in range(n_layers):
            pca = PCA(n_components=0.99)
            pca.fit(embeddings[:, layer])
            n_dim = pca.explained_variance_ratio_.shape[0] / initial_ndim * 100
            dims[layer] = n_dim

        print("Saving results")

        save_pickle(dims, ROOT / f'results/intrinsic_dimensionality/pca_{self.model_name}.pkl')

    
    def compute_id_grade(self):

        """
            Computes intrinsic dimensionality with the GRIDE algorithm
            and saves the results.
        """

        embeddings = self.load_embeddings()
        n_samples, n_layers, _ = embeddings.shape
        scale = round(np.log2(n_samples))

        results = np.empty((n_layers, scale))

        print("Computing results...")

        for layer in range(n_layers):
            data = Data(embeddings[:, layer])
            data.remove_identical_points()
            id_list, id_error_list, id_distance_list = data.return_id_scaling_gride(range_max=2**scale)
            results[layer] = id_list
            del data

        print("Saving results")

        save_pickle(results, ROOT / f'results/intrinsic_dimensionality/{self.model_name}.pkl')

    def load_pca_results(self):
        
        target_path = ROOT / f'results/intrinsic_dimensionality/pca_{self.model_name}.pkl'
        
        if not os.path.exists(target_path):
            print("Results not available, will be computed")
            self.compute_pca()

        results = open_pickle(target_path)

        return results

    def load_grade_results(self):
        
        target_path = ROOT / f'results/intrinsic_dimensionality/{self.model_name}.pkl'
        
        if not os.path.exists(target_path):
            print("Results not available, will be computed")
            self.compute_id_grade()

        results = open_pickle(target_path)

        return results
    
    def plot_pca(self):

        results = self.load_pca_results()
        fig, ax = plt.subplots(figsize=(5, 3.5), tight_layout=True)
        ax.grid(visible=True, axis='y')
        n_layers = results.shape[0]
        plt.plot(np.arange(n_layers), results, '.-')
        plt.title(self.names_to_titles[self.model_name])
        plt.ylabel('% Initial Dimensions')
        plt.xlabel("Layer")
        plt.show()


    def plot_scales(self):

        cmap = plt.get_cmap('autumn_r')

        fig, ax = plt.subplots(figsize=(5, 3.5), tight_layout=True)
        ax.grid(visible=True, axis='y')

        results = self.load_grade_results()
        n_layers, n_scales = results.shape

        # x = np.arange(n_scales)
        x = 906*6 / np.array([2**i for i in range(n_scales)])
        # print(x)

        # Create normalization based on layer index
        norm = mpl.colors.Normalize(vmin=1, vmax=n_layers)

        for layer in range(n_layers):
            ax.plot(
                x,
                results[layer],'.-',
                alpha=0.5,
                color=cmap(norm(layer + 1))
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$Log_{10}(\frac{n}{\operatorname{rank}(NN)})$"+" — Local to global")
        # Create ScalarMappable for colorbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Layer")

        ax.set_title(self.names_to_titles[self.model_name])

        plt.show()

    def plot_scales_gif(self, interval=500):

        results = self.load_grade_results()
        n_layers, n_scales = results.shape
        # x = np.arange(n_scales)
        x = 906*6 / np.array([2**i for i in range(n_scales)])


        cmap = plt.get_cmap("autumn_r")
        norm = mpl.colors.Normalize(vmin=1, vmax=n_layers)

        fig, ax = plt.subplots(figsize=(4, 3.5), tight_layout=True)
        ax.grid(visible=True, axis='y')
        ax.set_title(self.names_to_titles[self.model_name])

        # Fix axis limits so they don't rescale each frame
        # ax.set_xlim(0, n_scales - 1)
        ax.set_xlim(np.min(x[x > 0]), np.max(x))
        ax.set_ylim(np.min(results), np.max(results))
        

        lines = []

        # Initialize empty line objects (one per layer)
        for layer in range(n_layers):
            line, = ax.plot([], [], '.-', alpha=0.7,
                            color=cmap(norm(layer + 1)))
            lines.append(line)

        # Add colorbar (static)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Layer")
        ax.set_xscale("log")
        ax.set_xlabel(r"$Log_{10}(\frac{n}{\operatorname{rank}(NN)})$")
        
        cbar.set_ticks(np.arange(1, n_layers + 1))

        def update(frame):
            """
            Frame 0 -> show layer 0
            Frame 1 -> show layer 1
            ...
            """
            lines[frame].set_data(x, results[frame])
            return lines

        anim = FuncAnimation(
            fig,
            update,
            frames=n_layers,
            interval=interval,
            blit=True,
            repeat=False
        )

        gif_path = ROOT / f"results/intrinsic_dimensionality/plots_and_gifs/{self.model_name}.gif"
        anim.save(gif_path, writer=PillowWriter(fps=1000 // interval))
        plt.close(fig)

    def plot_embedder(self):

        fig, ax = plt.subplots(figsize=(5, 3.5), tight_layout=True)
        ax.grid(visible=True, axis='y')
        results = self.load_grade_results()
        n_layers, n_scales = results.shape
        x = np.arange(n_layers)
        y = results[x, self.best_scale_indices[self.model_name]]
        # y = results[x, [5]*n_layers]

        plt.plot(x, y, '.-')
        plt.title(self.names_to_titles[self.model_name])
        plt.ylabel("# Dimensions")
        plt.xlabel("Layer")
        plt.show()

    def plot_both_dimensionalities(self):

        id_results = self.load_grade_results()
        pca_results = self.load_pca_results()
        n_layers, n_scales = id_results.shape
        x = np.arange(n_layers)
        y = id_results[x, self.best_scale_indices[self.model_name]]

        fig, ax1 = plt.subplots(figsize=(5, 3.5), tight_layout=True)

        color = 'tab:red'
        ax1.set_xlabel('# Layers')
        ax1.grid(visible=True, axis='y')
        ax1.set_ylabel('Intrinsic dimensionality', color=color)
        ax1.plot(x, y, '.-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('PCA % Initial Dimensions', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, pca_results, '.-',color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(self.names_to_titles[self.model_name])
        plt.show()

