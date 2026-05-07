# What Makes Linguistic Representations Good Models of High-Level Visual Perception in the Human Brain?

This repository contains the research code for the paper _What Makes Linguistic Representations Good Models of High-Level Visual Perception in the Human Brain?_, by Anna Bavaresco, Ina Klarić, Raquel Fernández, and Sien Moens. 


<img width="3444" height="856" alt="exp_pipeline_1" src="https://github.com/user-attachments/assets/751e193c-2d48-4de7-b156-010f916be9fd" />

<img width="2905" height="2199" alt="exp_pipeline_2" src="https://github.com/user-attachments/assets/a68adbc5-d504-49d9-a456-943868da058d" />


The repository is structured as follows:

```
.
├── LICENSE
├── README.md
├── data
│   └── Image captions.xlsx
├── data_analyses
│   ├── brain_rsa
│   ├── caption_metrics
│   ├── encoding
│   └── simj_rsa
├── data_explore
│   ├── get_images_and_coco_captions.ipynb
│   └── simlarity_judgments.ipynb
├── job-scripts
│   ├── get_embeddings.sh
│   ├── get_model_captions.sh
│   └── run_rsa.sh
├── results
│   ├── lm_brain_rsa_results.csv
│   ├── lm_encoding.csv
│   ├── lm_judg_rsa_results.csv
│   └── visual_models_encoding.csv
├── setup.py
└── src
    ├── __init__.py
    ├── __pycache__
    ├── brain_encoding
    ├── caption_generation
    ├── caption_metrics
    ├── fmri_responses
    ├── indexing_and_formatting
    ├── paths.py
    ├── representation_extraction
    ├── rsa
    └── utils.py


```

## Data

The data analyses in our experiments come from two main sources:
1. The [Natural Scenes Dataset](https://www.nature.com/articles/s41593-021-00962-x), providing fMRI responses and behavioural judgments over multiple images;
2. The [MS COCO dataset](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48), introducing the images used as stimuli in the NSD experiments and crowdsourced image captions. 

The *fMRI responses* used in our study can be downloaded by running the file `src/fmri_responses/download_brain_data.py`.

The *behavioural similarity judgments* can be downloaded [here](https://natural-scenes-dataset.s3.amazonaws.com/nsddata/bdata/meadows/Meadows_nsd-multiple-arrangements_v_v2_tree.json).

The image stimuli shown to the participants in the NSD experiments are publicly available [here](https://natural-scenes-dataset.s3.amazonaws.com/nsddata/stimuli/nsd/shared1000/) and can be downloaded and matched with MS COCO captions using the code provided in `data_explore/get_images_and_coco_captions.ipynb`.

It is also useful to download [this file](https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_expdesign.mat), allowing to map between NSD and MS COCO indexing conventions. 


## Setup

To be able to run the code included in this repo, please structure your environment using the following commands:
```
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Reproducing Results

The main results from our paper can be reproduced by running the files contained in `job-scripts`. These scripts take care of the following steps:
1. Obtaining machine-generated image captions by running the code contained in `src/caption_generation`;
2. Embedding machine-generated and human-annotated image captions with different language models and computing image features using multiple vision models, running the code provided in `src/representation_extraction`;
3. Computing brain encoding results by running the scripts contained in `src/brain_encoding`;
4. Computing results from representational similarity analysis (RSA) on both neural and behavioural data (similarity judgments), using the code provided in `src/rsa`.

Additional code to analyse these results is provided in `data_analyses`. More specifically, the notebooks included there allow:
* Computing metrics (perplexity, lexical density, visualness) on different caption types (`data_analyses/caption_metrics/caption_metrics.ipynb`); 
* Visualising brain encoding results (`data_analyses/encoding/visualise_encoding_results.ipynb`);
* Visualising neural and behavioural RSA results (`data_analyses/brain_rsa/rsa.ipynb` and `data_analyses/simj_rsa/simj_rsa.ipynb`);
* Fitting mixed-effects models to brain encoding, neural RSA, and behavioural RSA results (`data_analyses/brain_rsa/statistical_analyses_brain_rsa.Rmd`, `data_analyses/encoding/statistical_analyses.Rmd`, and `data_analyses/simj_rsa/statistical_analyses_judg.Rmd`).

To facilitate the reproduction of plots and statistical analyses, we include a subset of our main results in `results`.

Additionally, we include all captions included in the experiments in `data/Image captions.xlsx`.
