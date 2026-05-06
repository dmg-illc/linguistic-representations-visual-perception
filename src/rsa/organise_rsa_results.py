from src.utils import *
from src.paths import ROOT
import pandas as pd

class BrainRSAResultsManager():

    def __init__(self):
        self.caption_types = {'coco': 'COCO', 'loc-narr': 'LocNarr', 'llava': 'LLaVA', 'phi': 'Phi-4', 
                              'pixtral': 'Pixtral', 'qwen': 'QwenVL', 'molmo': 'Molmo'}
                              
        self.caption_embedders = {'qwen3': {'template': 'qwen3-{cap_type}-last', 'name': 'Qwen3 Embeddings'},
                                'kalm': {'template': 'kalm-{cap_type}', 'name': 'Kalm Embeddings'},
                                'bert': {'template': 'bert-{cap_type}-cls', 'name': 'BERT'},
                                'llama': {'template': 'llama-{cap_type}-last', 'name': 'Llama3'},
                                'gpt2': {'template': 'gpt2-{cap_type}-last', 'name': 'GPT-2'}}

        self.rois = ['all', 'places', 'faces', 'bodies']

        self.dataframe = pd.DataFrame({'enc_name': [], 'caption_type': [], 'participant':[], 'roi': [], 'rs': []})


    def add_results(self):
        for embedder  in self.caption_embedders:
            for ctype in self.caption_types:
                res = open_pickle(ROOT / f"results/brain_rsa/{self.caption_embedders[embedder]['template'].format(cap_type = ctype)}.pkl")
                for roi in self.rois:
                    roi_res = res['rs'][roi].max(axis=0)
                    for part, rs in enumerate(roi_res):
                        self.dataframe.loc[len(self.dataframe)] = [self.caption_embedders[embedder]['name'], self.caption_types[ctype], part+1, roi, rs]
                
                del res

        print("Results added to dataframe")
        self.dataframe.to_csv(ROOT / "results/brain_rsa/lm_brain_rsa_results.csv", index=False)        
        print("Dataframe saved as CSV")

class Simj_RSAResultsManager():

    def __init__(self):
        self.caption_types = {'coco': 'COCO', 'loc-narr': 'LocNarr', 'llava': 'LLaVA', 'phi': 'Phi-4', 
                              'pixtral': 'Pixtral', 'qwen': 'QwenVL', 'molmo': 'Molmo'}
                              
        self.caption_embedders = {'qwen3': {'template': 'qwen3-{cap_type}-last', 'name': 'Qwen3 Embeddings'},
                                'kalm': {'template': 'kalm-{cap_type}', 'name': 'Kalm Embeddings'},
                                'bert': {'template': 'bert-{cap_type}-cls', 'name': 'BERT'},
                                'llama': {'template': 'llama-{cap_type}-last', 'name': 'Llama3'},
                                'gpt2': {'template': 'gpt2-{cap_type}-last', 'name': 'GPT-2'}}


        self.dataframe = pd.DataFrame({'enc_name': [], 'caption_type': [], 'participant':[], 'rs': []})


    def add_results(self):
        for embedder  in self.caption_embedders:
            for ctype in self.caption_types:
                res = open_pickle(ROOT / f"results/judj_rsa/{self.caption_embedders[embedder]['template'].format(cap_type = ctype)}.pkl")
                best_res = res['rs'].max(axis=0)
                for part, rs in enumerate(best_res):
                    self.dataframe.loc[len(self.dataframe)] = [self.caption_embedders[embedder]['name'], self.caption_types[ctype], part+1, rs]
                
                del res

        print("Results added to dataframe")
        self.dataframe.to_csv(ROOT / "results/judj_rsa/lm_judg_rsa_results.csv", index=False)        
        print("Dataframe saved as CSV")

brain_rm = BrainRSAResultsManager()
brain_rm.add_results()

simj_rm = Simj_RSAResultsManager()
simj_rm.add_results()