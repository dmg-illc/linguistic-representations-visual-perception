'''
Here we'll create one single mask that extracts all voxels
belonging to scene-selective, body-selective, and face-selective
ROIs. Each participant will have a specific mask. 
'''

from src.paths import ROOT
from src.utils import *
from urllib.request import urlretrieve
import os
import nibabel as nib
import numpy as np

if __name__ == "__main__":

    for participant in range(1, 9):

        target_path = ROOT / f'data/fmri/subj0{participant}/rois/all_floc_rois.pkl'
        
        if not os.path.exists(target_path):

            part_mask = {}

            for hemisphere in ['lh', 'rh']:
                
                seg_places = nib.load(ROOT / f'data/fmri/subj0{participant}/rois/{hemisphere}.floc-places.mgz').get_fdata().squeeze()
                seg_faces = nib.load(ROOT / f'data/fmri/subj0{participant}/rois/{hemisphere}.floc-faces.mgz').get_fdata().squeeze()
                seg_bodies = nib.load(ROOT / f'data/fmri/subj0{participant}/rois/{hemisphere}.floc-bodies.mgz').get_fdata().squeeze()
                seg = np.zeros(seg_places.shape)
                seg[(seg_places != 0) | (seg_faces != 0) | (seg_bodies != 0)] = 1
                part_mask[hemisphere] = seg
            
            save_pickle(part_mask, target_path)

        