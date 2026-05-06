from src.paths import ROOT
from urllib.request import urlretrieve
import argparse
import os
from src.indexing_and_formatting.image_indexing_utils import pad_session_id

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nsessions")
parser.add_argument("-s", "--subject", choices=['1', '2', '3', '4', '5', '6', '7', '8'])
args = parser.parse_args()


def download_atlas_mapping(subject_id: int, hemisphere: str):
    url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj0{subject_id}/transforms/{hemisphere}.white-to-fsaverage.mgz"
    (ROOT / f"data/fmri/subj0{subject_id}/mapping").mkdir(parents=True, exist_ok=True)
    path = ROOT / f"data/fmri/subj0{subject_id}/mapping/{hemisphere}.white-to-fsaverage.mgz"
    if not os.path.exists(path):
        urlretrieve(url, path)

def download_noise_ceiling(subject_id: int, hemisphere: str):
    url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj0{subject_id}/nativesurface/betas_fithrf_GLMdenoise_RR/{hemisphere}.ncsnr.mgh"
    (ROOT / f"data/fmri/subj0{subject_id}/noise_ceilings").mkdir(parents=True, exist_ok=True)
    path = ROOT / f"data/fmri/subj0{subject_id}/noise_ceilings/{hemisphere}.ncsnr.mgh"
    if not os.path.exists(path):
        urlretrieve(url, path)


def download_session_fmri_data(subject_id: int, hemisphere: str, session_id: str):

    url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj0{subject_id}/nativesurface/betas_fithrf_GLMdenoise_RR/{hemisphere}.betas_session{session_id}.hdf5"
    path = ROOT / f"data/fmri/subj0{subject_id}/{hemisphere}/{hemisphere}_betas_session{session_id}.hdf5"
    if not os.path.exists(path):
        urlretrieve(url, path)

def download_roi_mapping(subject_id: int, hemisphere: str):
    (ROOT / f"data/fmri/subj0{subject_id}/rois").mkdir(parents=True, exist_ok=True)
    target_rois = ['floc-places', 'floc-bodies', 'floc-faces']
    for rois in target_rois:
        url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/freesurfer/subj0{subject_id}/label/{hemisphere}.{rois}.mgz"
        path = ROOT / f"data/fmri/subj0{subject_id}/rois/{hemisphere}.{rois}.mgz"
        if not os.path.exists(path):
            urlretrieve(url, path)



def download_subject_data(subject_id: int, n_sessions: int):
    (ROOT / f"data/fmri/subj0{subject_id}").mkdir(parents=True, exist_ok=True)

    
    download_roi_mapping(subject_id=subject_id, hemisphere='r')
    download_roi_mapping(subject_id=subject_id, hemisphere='l')

    download_noise_ceiling(subject_id=subject_id, hemisphere='r')
    download_noise_ceiling(subject_id=subject_id, hemisphere='l')

    download_atlas_mapping(subject_id=subject_id, hemisphere='r')
    download_atlas_mapping(subject_id=subject_id, hemisphere='l')

    for session in range(1, n_sessions+1):
        padded_session_id = pad_session_id(session)
        (ROOT / f"data/fmri/subj0{subject_id}/rh").mkdir(parents=True, exist_ok=True)
        download_session_fmri_data(subject_id=subject_id, hemisphere='r', session_id=padded_session_id)
        (ROOT / f"data/fmri/subj0{subject_id}/lh").mkdir(parents=True, exist_ok=True)
        download_session_fmri_data(subject_id=subject_id, hemisphere='l', session_id=padded_session_id)

    

if __name__ == "__main__":

    download_subject_data(subject_id=args.subject, n_sessions=int(args.nsessions))

    