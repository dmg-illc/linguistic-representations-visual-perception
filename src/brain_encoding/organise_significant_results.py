from src.paths import ROOT 
from src.utils import *
import pandas as pd
from src.brain_encoding.manage_encoding_results import ParticipantResults


def create_lm_results_csv():

    """
    This function will create and save a csv containing the significant results (only
    best layer per voxel) obtained with caption embeddings broken down per participant, ROI, caption type,
    and caption embedding. 
    """

    dataframe = pd.DataFrame({'enc_name': [], 'caption_type': [], 'participant':[], 'roi': [], 'acc': []})
    model_templates = ['qwen3-{cap_type}-last', 'gpt2-{cap_type}-last', 'llama-{cap_type}-last', 'bert-{cap_type}-cls', 'kalm-{cap_type}']
    model_names = ['Qwen3 Embeddings', 'GPT-2', 'Llama3', 'BERT', 'Kalm Embeddings']
    caption_types = ['coco', 'coco-avg', 'llava', 'phi', 'pixtral', 'qwen', 'molmo']
    caption_names = ['COCO', 'COCO avg.', 'LLaVA-OV','Phi-4','Pixtral', 'Qwen2.5-VL', 'Molmo']

    for m, model_template in enumerate(model_templates):
        for i, ctype in enumerate(caption_types):
            print(ctype)
            res = open_pickle(ROOT / f'results/encoding/encoding_by_model/{model_template.format(cap_type=ctype)}.pkl')
            masks = open_pickle(ROOT / f'results/encoding/significance_masks/{model_template.format(cap_type=ctype)}.pkl')
            for participant in range(1,9):
                # print(i)
                p_results = ParticipantResults(results=res, masks=masks, participant=participant)
                for target_roi in ['all', 'faces', 'bodies', 'places']:
                    p_acc = p_results.get_roi_accuracies_avg(target_roi)
                    dataframe.loc[len(dataframe)] = [model_names[m], caption_names[i], participant, target_roi, p_acc]
                    del p_acc
                del p_results

    dataframe.to_csv(ROOT / 'results/encoding/encoding_by_model' / 'all_rois-results.csv', index=False)
        

def create_vf_results_csv():

    """
    This function will create and save a csv containing the significant encoding results (only
    best layer per voxel) obtained with visual features broken down per participant, ROI, architecture,
    and training regime. 
    """

    dataframe = pd.DataFrame({'model_name': [], 'training_type': [], 'participant':[], 'roi': [], 'acc': []})
    model_templates = ['resnet{train_type}', 'vit{train_type}']
    model_names = ['ResNet50', 'ViT']
    training_types = ['', '-clip', '-dino']
    training_labels = ['ImageNet', 'CLIP', 'DINO']
    
    for m, model_template in enumerate(model_templates):
        for i, ttype in enumerate(training_types):
            res = open_pickle(ROOT / f'results/encoding/encoding_by_model/{model_template.format(train_type=ttype)}.pkl')
            masks = open_pickle(ROOT / f'results/encoding/significance_masks/{model_template.format(train_type=ttype)}.pkl')
            for participant in range(1,9):
                p_results = ParticipantResults(results=res, masks=masks, participant=participant)
                for target_roi in ['all', 'faces', 'bodies', 'places']:
                    p_acc = p_results.get_roi_accuracies_avg(target_roi)
                    dataframe.loc[len(dataframe)] = [model_names[m], training_labels[i], participant, target_roi, p_acc]
                    del p_acc
                del p_results

    dataframe.to_csv(ROOT / 'results/encoding/encoding_by_model' / 'all_rois-results-visual.csv', index=False)

create_lm_results_csv()
create_vf_results_csv()