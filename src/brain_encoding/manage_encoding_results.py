import nibabel as nib
from src.paths import ROOT
import numpy as np
from src.utils import *
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from src.fmri_responses.fmri_loading import ParticipantResponses
# from src.brain_encoding.intrinsic_dimensionality import IDAnalysis


class ParticipantResults():
    """
    This class is useful for managing encoding results, along with their significance masks. 
    """

    def __init__(self, results: dict, participant: int, significant_results_only = True, masks=None):
        
        self.participant = participant
        self.part_results = results[str(self.participant)]['all']
        self.n_layers = self.part_results['lh'].shape[1]
        if masks != None:
            self.signif_masks = masks[str(self.participant)]
        self.signif_only = significant_results_only
        self.all_seg_mask = open_pickle(ROOT / f'data/fmri/subj0{self.participant}/rois/all_floc_rois.pkl')
        self.rois = {'places': {'OPA': 1, 'PPA': 2, 'RSC': 3}, 
                    'faces': {'OFA': 1, 'FFA-1': 2, 'FFA-2': 3, 'mTL-faces': 4, 'aTL-faces': 5}, 
                    'bodies': {'EBA': 1, 'FBA-1': 2, 'FBA-2': 3, 'mTL-bodies': 4}, 'all': {}}

    def get_roi_accuracies(self, roi: str, subroi = None):

        """
        Returns an array of bilateral encoding accuracies by filtering the voxels of 
        the specified ROI or sub-ROI. Additionally, it applies significance masks
        to ensure only accuracies computed on significant voxels are returned. 
        """
        
        # validating arguments
        if roi not in self.rois:
            raise ValueError(f"Invalid ROI! Valid ROIs include: {', '.join(self.rois)}")
        
        if subroi:
            if subroi not in self.rois[roi]:
                raise ValueError("Invalid subROI!")

        # extracting best (across layers) roi accuracies for both hemispheres
        lh_acc = self.part_results['lh'].mean(axis = 0).max(axis=0)
        rh_acc = self.part_results['rh'].mean(axis = 0).max(axis=0)

        if self.signif_only:
            # setting non significant voxels to implausible value 
            lh_acc[~self.signif_masks['lh']] = -20
            rh_acc[~self.signif_masks['rh']] = -20

        # mapping accuracies to their actual voxels and filtering significant ones
        lh_acc_mapped = np.zeros_like(self.all_seg_mask['lh'])
        lh_acc_mapped[(self.all_seg_mask['lh'] != 0)] = lh_acc

        rh_acc_mapped = np.zeros_like(self.all_seg_mask['rh'])
        rh_acc_mapped[(self.all_seg_mask['rh'] != 0)] = rh_acc

        # if we want all rois, we can extract all the relevant voxels and apply significance masks
        if roi == 'all':
            roi_mask_lh, roi_mask_rh = self.all_seg_mask['lh'] != 0, self.all_seg_mask['rh'] != 0
            to_ret = np.concatenate((lh_acc_mapped[roi_mask_lh], rh_acc_mapped[roi_mask_rh]))
            return to_ret[to_ret!=-20]
        
        # if we want to focus on a specific ROI, we need to apply both ROI mask and significance masks
        else:
            lh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/lh.floc-{roi}.mgz').get_fdata()[:,0,0]
            rh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/rh.floc-{roi}.mgz').get_fdata()[:,0,0]

            if subroi:
                lh_target_accs = lh_acc_mapped[lh_seg == self.rois[roi][subroi]]
                rh_target_accs = rh_acc_mapped[rh_seg == self.rois[roi][subroi]]
            
            else:
                lh_target_accs = lh_acc_mapped[lh_seg != 0]
                rh_target_accs = rh_acc_mapped[rh_seg != 0]
            
            return np.concatenate((lh_target_accs[lh_target_accs != -20], rh_target_accs[rh_target_accs != -20]))
            
    def get_roi_accuracies_fsaverage(self, roi: str, subroi = None):

        """
        Returns an array of bilateral encoding accuracies by filtering the voxels of 
        the specified ROI or sub-ROI. Additionally, it applies significance masks
        to ensure only accuracies computed on significant voxels are returned. 
        """
        
        # validating arguments
        if roi not in self.rois:
            raise ValueError(f"Invalid ROI! Valid ROIs include: {', '.join(self.rois)}")
        
        if subroi:
            if subroi not in self.rois[roi]:
                raise ValueError("Invalid subROI!")

        # extracting best (across layers) roi accuracies for both hemispheres
        lh_acc = self.part_results['lh'].mean(axis = 0).max(axis=0)
        rh_acc = self.part_results['rh'].mean(axis = 0).max(axis=0)

        if self.signif_only:
            # setting non significant voxels to implausible value 
            lh_acc[~self.signif_masks['lh']] = -20
            rh_acc[~self.signif_masks['rh']] = -20

        # mapping accuracies to their actual voxels and filtering significant ones
        lh_acc_mapped = np.zeros_like(self.all_seg_mask['lh'])
        lh_acc_mapped[(self.all_seg_mask['lh'] != 0)] = lh_acc

        rh_acc_mapped = np.zeros_like(self.all_seg_mask['rh'])
        rh_acc_mapped[(self.all_seg_mask['rh'] != 0)] = rh_acc

        # loading mapping to fsaverage
        native_to_fsaverage_mapping_lh = nib.load(ROOT / f'data/fmri/subj0{self.participant}/mapping/lh.white-to-fsaverage.mgz').get_fdata()
        native_to_fsaverage_mapping_rh = nib.load(ROOT / f'data/fmri/subj0{self.participant}/mapping/rh.white-to-fsaverage.mgz').get_fdata()

        # mapping to fsaverage
        fsaverage_acc_lh = lh_acc_mapped[np.squeeze(native_to_fsaverage_mapping_lh.astype(int)) - 1]
        fsaverage_acc_rh = rh_acc_mapped[np.squeeze(native_to_fsaverage_mapping_rh.astype(int)) - 1]

        # concatenating bilateral responses
        to_ret = np.concatenate((fsaverage_acc_lh, fsaverage_acc_rh))
        
        # resetting non-significant values to 0
        to_ret[to_ret==-20] = 0

        del native_to_fsaverage_mapping_lh, native_to_fsaverage_mapping_rh
        return to_ret


    def get_roi_accuracies_all_layers(self, roi: str, subroi = None):
        """
        Returns a n_layers x n_voxels matrix of voxel-wise accuracies. Applies ROI masks but NOT
        significance masks.
        """

        # validating arguments
        if roi not in self.rois:
            raise ValueError(f"Invalid ROI! Valid ROIs include: {', '.join(self.rois)}")
        
        if subroi:
            if subroi not in self.rois[roi]:
                raise ValueError("Invalid subROI!")

        # extracting target layer accuracies for both hemispheres
        lh_acc = self.part_results['lh'].mean(axis = 0)
        rh_acc = self.part_results['rh'].mean(axis = 0)

        # mapping accuracies to their actual voxels 
        lh_acc_mapped = np.zeros((self.n_layers, self.all_seg_mask['lh'].shape[0]))
        lh_acc_mapped[:, (self.all_seg_mask['lh'] != 0)] = lh_acc

        rh_acc_mapped = np.zeros((self.n_layers, self.all_seg_mask['rh'].shape[0]))
        rh_acc_mapped[:, (self.all_seg_mask['rh'] != 0)] = rh_acc

        # applying ROI masks and concatenating voxels from both hemispheres
        if roi == 'all':
            roi_mask_lh, roi_mask_rh = self.all_seg_mask['lh'] != 0, self.all_seg_mask['rh'] != 0
            to_ret = np.concatenate((lh_acc_mapped[:,roi_mask_lh], rh_acc_mapped[:,roi_mask_rh]), axis=1)
            return to_ret.mean(axis=1)
        
        else:
            lh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/lh.floc-{roi}.mgz').get_fdata()[:,0,0]
            rh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/rh.floc-{roi}.mgz').get_fdata()[:,0,0]

          
            lh_target_accs = lh_acc_mapped[:, (lh_seg != 0)]
            rh_target_accs = rh_acc_mapped[:, (rh_seg != 0)]
            
            return np.concatenate((lh_target_accs, rh_target_accs), axis=1).mean(axis=1)
    
    def get_roi_accuracies_avg(self, roi: str, subroi = None):

        """
        Returns average ROI accuracy, considering the best layer for each voxel and 
        filtering only significant voxels.
        """

        return self.get_roi_accuracies(roi, subroi).mean().item()

    def get_roi_best_layers(self, roi: str, subroi = None):

        """
        Returns an array indicating the index of the most predictive layer for each voxel
        belonging to the specified ROI.
        Only significant voxels are considered.
        """
        
        # validating arguments
        if roi not in self.rois:
            raise ValueError(f"Invalid ROI! Valid ROIs include: {', '.join(self.rois)}")
        
        if subroi:
            if subroi not in self.rois[roi]:
                raise ValueError("Invalid subROI!")

        # extracting best layers for both hemispheres
        lh_bl = self.part_results['lh'].mean(axis = 0).argmax(axis=0)
        rh_bl = self.part_results['rh'].mean(axis = 0).argmax(axis=0)

        # setting best layers from non-significant voxels to implausible value 
        lh_bl[~self.signif_masks['lh']] = -20
        rh_bl[~self.signif_masks['rh']] = -20

        # mapping best layers to their actual voxels and filtering significant ones
        lh_bl_mapped = np.zeros_like(self.all_seg_mask['lh'])
        lh_bl_mapped[(self.all_seg_mask['lh'] != 0)] = lh_bl

        rh_bl_mapped = np.zeros_like(self.all_seg_mask['rh'])
        rh_bl_mapped[(self.all_seg_mask['rh'] != 0)] = rh_bl

        # applying ROI masks and returning best layers
        if roi == 'all':
            roi_mask_lh, roi_mask_rh = self.all_seg_mask['lh'] != 0, self.all_seg_mask['rh'] != 0
            to_ret = np.concatenate((lh_bl_mapped[roi_mask_lh], rh_bl_mapped[roi_mask_rh]))
            return to_ret[to_ret!=-20]
        
        else:
            lh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/lh.floc-{roi}.mgz').get_fdata()[:,0,0]
            rh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/rh.floc-{roi}.mgz').get_fdata()[:,0,0]

            if subroi:
                lh_target_accs = lh_bl_mapped[lh_seg == self.rois[roi][subroi]]
                rh_target_accs = rh_bl_mapped[rh_seg == self.rois[roi][subroi]]
            
            else:
                lh_target_accs = lh_bl_mapped[lh_seg != 0]
                rh_target_accs = rh_bl_mapped[rh_seg != 0]
            
            return np.concatenate((lh_target_accs[lh_target_accs != -20], rh_target_accs[rh_target_accs != -20]))



class BrainPlotManager():
    
    """
    Class that creates all the plots useful to visualise brain encoding results. 
    """
    
    def __init__(self):
        self.caption_types = {'coco': 'MS COCO', 'llava': 'LLaVA-OV', 'phi': 'Phi-4', 
                              'pixtral': 'Pixtral', 'qwen': 'Qwen2.5-VL', 'molmo': 'Molmo'}
        self.hatches = {'coco': '', 'llava': '..', 'phi': '//', 
                              'pixtral': '++', 'qwen': '\\\\//', 'molmo': '//..', 'coco-avg': '++..'}
        self.caption_embedders = {'qwen3': {'template': 'qwen3-{cap_type}-last', 'name': 'Qwen3 Embedding'},
                                'kalm': {'template': 'kalm-{cap_type}', 'name': 'KaLM Embedding'},
                                'bert': {'template': 'bert-{cap_type}-cls', 'name': 'BERT'},
                                'llama': {'template': 'llama-{cap_type}-last', 'name': 'Llama3'},
                                'gpt2': {'template': 'gpt2-{cap_type}-last', 'name': 'GPT-2'}}
        self.brain_rois = ['faces', 'bodies', 'places', 'all']
        self.visual_features = {'vit': 'ViT', 'resnet': 'ResNet50'}
        self.training_types = {'-dino': 'DINO', '': 'ImageNet', '-clip': 'CLIP'}

    
    def comprehensive_lm_plot_by_encoder(self, target_roi: str, save=False, file_name = ''):
        
        """
        Creates one accuracy barplot with all ROIs, organised by caption types. 
        It averages participant-wise accuracies and shows std across participant
        as error bars. 
        """

        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")
        
        dataframe = pd.read_csv(ROOT / 'results/lm_encoding.csv') 
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(6,3), tight_layout=True)
        plt.grid(visible=True, axis='y')
        spacing_parameter = 0.14
        # spacing_parameter = 0.13

        for i, emb in enumerate(self.caption_embedders):
            for j, cn in enumerate(self.caption_types):
            
                emb_name = self.caption_embedders[emb]['name']
                y = dataframe.loc[(dataframe.caption_type==self.caption_types[cn]) & (dataframe.roi == target_roi) & (dataframe.enc_name == emb_name), 'acc'].values
                
                plt.bar(i+spacing_parameter*j, y.mean(),yerr=y.std(), color = cmap(j), width = spacing_parameter, alpha=0.8, hatch = self.hatches[cn], edgecolor='white')
           
        legend_elements = [Patch(facecolor=cmap(i),
                            label=self.caption_types[ct], hatch=self.hatches[ct], edgecolor='white') for i, ct in enumerate(self.caption_types)]
        plt.legend(handles=legend_elements, handleheight=1.2, ncols=3, loc='upper right')
        plt.ylim(0.2,0.55)
        emb_names = [self.caption_embedders[emb]['name'].replace(' ', '\n') for emb in self.caption_embedders]
        plt.xticks(np.arange(len(self.caption_embedders))+3*spacing_parameter, emb_names)
        plt.title(f"Caption Embeddings — ROI: {target_roi.capitalize()}")
        plt.ylabel("Pearson's r")
        plt.gca().set_axisbelow(True)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")
            
        plt.show()
        del dataframe
        
    def vf_plot(self, target_roi: str, save=False, file_name = ''):
        
        """
        Creates one accuracy barplot for vision models, organised by model architecture. 
        It averages participant-wise accuracies and shows std across participant
        as error bars. 
        """
        
        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")
        
        dataframe = pd.read_csv(ROOT / 'results/visual_models_encoding.csv')
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(3,3), tight_layout=True)
        plt.grid(visible=True, axis='y')
        spacing_parameter = 0.25
        hatches = ['..', '//', '']
        for i, mod in enumerate(self.visual_features):
            for j, tn in enumerate(self.training_types):
                
                y = dataframe.loc[(dataframe.training_type==self.training_types[tn]) & (dataframe.roi == target_roi) & (dataframe.model_name == self.visual_features[mod]), 'acc'].values
                
                plt.bar(i+spacing_parameter*j, y.mean(),yerr=y.std(), color = cmap(j+6), width = spacing_parameter, alpha=0.8, hatch=hatches[j], edgecolor='white')
            
        legend_elements = [Patch(facecolor=cmap(i+6),
                            label=self.training_types[tt], hatch=hatches[i], edgecolor='white') for i, tt in enumerate(self.training_types)]
        
        plt.legend(handles=legend_elements,handleheight=1.2, ncols=2, loc='upper right')
        plt.ylim(0.2,0.55)
        model_names = [self.visual_features[mod]+'\n' for mod in self.visual_features]
        plt.xticks(np.arange(len(model_names))+spacing_parameter, model_names)
        plt.title(f"Visual Features — ROI: {target_roi.capitalize()}")
        plt.ylabel("Pearson's r")
        plt.gca().set_axisbelow(True)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf", bbox_inches='tight')
            # plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")

        plt.show()
        del dataframe

    def linechart_layerwise_accuracies(self, embedder: str, target_roi: str, save=False, file_name = ''):
        
        """
        Plots a linechart where every line indicates how the accuracy of a caption embedder
        progresses through the layers. Accuracies are shown for a specific ROI
        and one specific caption embedder. 
        """
        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")
        
        if embedder not in self.caption_embedders:
            raise ValueError("Enter a valid caption embedder!")
        
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(4.5,3.5), tight_layout=True)
        plt.grid(visible=True, axis='y')
        embedder_template = self.caption_embedders[embedder]['template'] 
        
        for i, caption_type in enumerate(self.caption_types):

            # loading results for the relevant caption encoder
            res = open_pickle(ROOT / f'results/encoding/encoding_by_model/{embedder_template.format(cap_type = caption_type)}.pkl')
            # creating list where we can store results for every participant
            part_accuracies = []
            
            # getting all-layer accuracies for each participant
            for participant in range(1,9):
                p_results = ParticipantResults(results=res, participant=str(participant))
                p_layer_acc = p_results.get_roi_accuracies_all_layers(roi=target_roi)
                part_accuracies.append(p_layer_acc)

            # plotting average across participants
            plt.plot(np.arange(p_results.n_layers), np.stack(part_accuracies).mean(axis=0), '.-', color=cmap(i), label=self.caption_types[caption_type], alpha=0.7)
            del res
        if embedder == 'bert' and target_roi == 'faces':    
            plt.legend(ncol=2)
        plt.ylim(0,0.36)
        plt.ylabel("Pearson's r")
        plt.xlabel("Layer")
        title = f"{self.caption_embedders[embedder]['name']} — ROI: {target_roi.capitalize()}"
        plt.title(title)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")
            
        plt.show() 

    def linechart_intrinsic_dimensionality(self, embedder: str, target_roi: str, axspan=None, save=False, file_name = ''):

        if target_roi not in self.brain_rois:
            raise ValueError("Enter a valid brain ROI!")
        
        if embedder not in self.caption_embedders:
            raise ValueError("Enter a valid caption embedder!")

        brain_accuracies = []
        embedder_template = self.caption_embedders[embedder]['template'] 

        for i, caption_type in enumerate(self.caption_types):

            # loading results for the relevant caption encoder
            res = open_pickle(ROOT / f'results/encoding/encoding_by_model/{embedder_template.format(cap_type = caption_type)}.pkl')
            # creating list where we can store results for every participant
            part_accuracies = []
            
            # getting all-layer accuracies for each participant
            for participant in range(1,9):
                p_results = ParticipantResults(results=res, participant=str(participant))
                p_layer_acc = p_results.get_roi_accuracies_all_layers(roi=target_roi)
                part_accuracies.append(p_layer_acc)
            
            brain_accuracies.append(np.stack(part_accuracies).mean(axis=0))


        # loading all relevant data            
        brain_acc = np.stack(brain_accuracies).mean(axis=0)
        id_analysis = IDAnalysis(embedder)
        id_results = id_analysis.load_grade_results()
        n_layers, n_scales = id_results.shape
        x = np.arange(n_layers)
        y = id_results[x, id_analysis.best_scale_indices[id_analysis.model_name]]

        fig, ax1 = plt.subplots(figsize=(4.5, 3), tight_layout=True)

        color = 'tab:red'
        ax1.set_xlabel('# Layers')
        ax1.grid(visible=True, axis='y')
        ax1.set_ylabel('Encoding Accuracy (r)', color=color)
        ax1.plot(x, brain_acc, '.-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Intrinsic dimensionality', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, y, '.-',color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        if axspan != None:
            ax2.axvspan(axspan[0], axspan[1], color="tab:red", alpha=0.1)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        title = f"{self.caption_embedders[embedder]['name']} — ROI: {target_roi.capitalize()}"
        plt.title(title)

        if save:
            plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.pdf")
            
        plt.show()
    
    def noise_ceilings_plot(self, caption_embedder: str, participant: int):

        if caption_embedder not in self.caption_embedders:
            raise ValueError("Invalid caption embedder!")
        
        if participant not in np.arange(1, 9):
            raise ValueError("Invalid participant!")
        
        # get accuracy averaged across caption types
        accs = []
        for caption_type in self.caption_types:
            model_id = self.caption_embedders[caption_embedder]['template'].format(cap_type=caption_type)
            res = open_pickle(ROOT / f'results/encoding/encoding_by_model/{model_id}.pkl')
            p_results = ParticipantResults(results=res, participant=str(participant), significant_results_only=False)
            acc = p_results.get_roi_accuracies(roi='all')
            accs.append(acc)
            del res, p_results

        avg_acc = np.stack(accs).mean(axis=0)

        # get explainable variance
        p_noise = ParticipantResponses(participant, noise_ceilings=True, activations=False)
        roi_noise_lh = p_noise.get_roi_noise_ceiling(roi='all', hemisphere='lh')
        roi_noise_rh = p_noise.get_roi_noise_ceiling(roi='all', hemisphere='rh')
        roi_noise = np.concatenate((roi_noise_lh, roi_noise_rh))

        # create plot
        plt.figure(figsize=(3.5,3), tight_layout=True)
        plt.grid(visible=True, axis='y')
        plt.scatter(roi_noise*100, avg_acc,s=2, alpha=0.3)
        plt.xlim(-5, 100)
        plt.ylim(-0.05, 1)
        plt.ylabel('Encoding accuracy (r)')
        plt.xlabel('% Explainable Variance')
        model_name = self.caption_embedders[caption_embedder]['name']
        plt.title(f"{model_name} – P{participant}")
        file_name = f"{model_name.split(' ')[0]}_{participant}"
        plt.savefig(ROOT / f"/projects/prjs1701/scene-captions/results/plots/{file_name}.png", dpi=300)
        plt.show()
        del p_noise, roi_noise_lh, roi_noise_rh, avg_acc



