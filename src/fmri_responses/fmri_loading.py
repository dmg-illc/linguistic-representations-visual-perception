import nibabel as nib
from scipy.io import loadmat
from src.paths import ROOT
import numpy as np
from src.utils import *


class ParticipantResponses():
    
    def __init__(self, participant: int, noise_ceilings=False, activations=True):

        self.participant = participant
        self.noise_ceilings = noise_ceilings
        self.rois = {'faces': {'OFA': 1, 'FFA-1': 2, 'FFA-2': 3, 'mTL-faces': 4, 'aTL-faces': 5},
                     'bodies': {'EBA': 1, 'FBA-1': 2, 'FBA-2': 3, 'mTL-bodies': 4},
                     'places': {'OPA': 1, 'PPA': 2, 'RSC': 3},
                     'all': {}}
        if activations:

            # loading all betas for the target participant
            self.lh_activations = open_pickle(ROOT / f'data/fmri/subj0{participant}/betas_shared1k/subj0{participant}_lh_shared1k_betas.pkl')
            self.rh_activations = open_pickle(ROOT / f'data/fmri/subj0{participant}/betas_shared1k/subj0{participant}_rh_shared1k_betas.pkl')
            self.image_ids = list(self.lh_activations.keys())
        
        
        if self.noise_ceilings:
            self.lh_noise_ceiling = self.adjust_noise_ceilings(nib.load(ROOT / f"data/fmri/subj0{participant}/noise_ceilings/lh.ncsnr.mgh").get_fdata()[:,0,0])
            self.rh_noise_ceiling = self.adjust_noise_ceilings(nib.load(ROOT / f"data/fmri/subj0{participant}/noise_ceilings/rh.ncsnr.mgh").get_fdata()[:,0,0])

    def get_roi_masks(self, roi: str): 


        if roi == 'all':
            seg = open_pickle(ROOT / f'data/fmri/subj0{self.participant}/rois/all_floc_rois.pkl')
            lh_mask, rh_mask = seg['lh'].astype(bool), seg['rh'].astype(bool)

        elif roi in self.rois:
            lh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/lh.floc-{roi}.mgz').get_fdata()[:,0,0]
            rh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/rh.floc-{roi}.mgz').get_fdata()[:,0,0]
            rh_mask, lh_mask = rh_seg != 0, lh_seg != 0

        else: 
            for high_level_roi in self.rois:
                if roi in self.rois[high_level_roi]:
                    lh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/lh.floc-{high_level_roi}.mgz').get_fdata()[:,0,0]
                    rh_seg = nib.load(ROOT / f'data/fmri/subj0{self.participant}/rois/rh.floc-{high_level_roi}.mgz').get_fdata()[:,0,0]
                    target_index = self.rois[high_level_roi][roi]
                    rh_mask, lh_mask = rh_seg ==target_index, lh_seg == target_index

        return lh_mask, rh_mask



    def get_roi_activations(self, roi: str, hemisphere: str): 

        lh_mask, rh_mask = self.get_roi_masks(roi)
        
        if hemisphere == 'lh':
            activations_dict = {img_id: self.lh_activations[img_id][lh_mask] for img_id in self.image_ids}
        
        elif hemisphere == 'rh':
            activations_dict = {img_id: self.rh_activations[img_id][rh_mask] for img_id in self.image_ids}
        
        elif hemisphere == 'both':
            activations_dict_lh = {img_id: self.lh_activations[img_id][lh_mask] for img_id in self.image_ids}
            activations_dict_rh = {img_id: self.rh_activations[img_id][rh_mask] for img_id in self.image_ids}
            activations_dict = {img_id: np.concatenate([activations_dict_lh[img_id], activations_dict_rh[img_id]]) for img_id in self.image_ids}

        return activations_dict
    
    def adjust_noise_ceilings(self, noise_ceiling: np.array):

        """
        Adjusts the noise ceilings accounting for the fact that not all images
        have been seen for the same number of times. 

        Code adapted from some of the NSD materials.
        """


        wh = [40, 40, 32, 30, 40, 32, 40, 30]
        exp_info = loadmat(ROOT / 'data/images/nsd_expdesign.mat', simplify_cells=True)
        temp = exp_info['masterordering'][:750 * wh[self.participant-1]]

        relevant_images =  []
        for i in range(1, 1001):
            relevant_images.append((temp==i).sum().item())

        rel_img_vec = np.array(relevant_images)
        a, b, c = (rel_img_vec==3).sum(), (rel_img_vec==2).sum(), (rel_img_vec==1).sum()

        adj_noise_ceil = noise_ceiling**2 / ( noise_ceiling**2 + ( (a/3 + b/2 + c) / (a + b + c) )  )

        return adj_noise_ceil

    def get_roi_noise_ceiling(self, roi: str, hemisphere: str):

        if not self.noise_ceilings:
            raise Exception("To be able to call this function, you need to set the noise_ceilings attribute to True")
        

        lh_mask, rh_mask = self.get_roi_masks(roi)
        
        if hemisphere == 'lh':
            roi_nc = self.lh_noise_ceiling[lh_mask]
        
        elif hemisphere == 'rh':
            roi_nc = self.rh_noise_ceiling[rh_mask]

        elif hemisphere == 'both':
            roi_nc_lh = self.lh_noise_ceiling[lh_mask]
            roi_nc_rh = self.rh_noise_ceiling[rh_mask]
            roi_nc = np.concatenate((roi_nc_lh, roi_nc_rh))
            
        return roi_nc
    