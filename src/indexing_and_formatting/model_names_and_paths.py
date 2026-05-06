from src.paths import ROOT


encoders_to_embeddings = {'qwen3': ['qwen3-molmo-last' , 'qwen3-coco-last', 'qwen3-llava-last', 'qwen3-phi-last', 'qwen3-pixtral-last', 'qwen3-qwen-last'],
                           'llama': ['llama-molmo-last' ,'llama-coco-last', 'llama-llava-last', 'llama-phi-last', 'llama-pixtral-last', 'llama-qwen-last'],
                           'bert': ['bert-molmo-cls' ,'bert-coco-cls', 'bert-llava-cls', 'bert-phi-cls', 'bert-pixtral-cls', 'bert-qwen-cls'],
                           'gpt2': ['gpt2-molmo-last' ,'gpt2-coco-last', 'gpt2-llava-last', 'gpt2-phi-last', 'gpt2-pixtral-last', 'gpt2-qwen-last'],
                           'kalm': ['kalm-coco', 'kalm-llava', 'kalm-phi', 'kalm-qwen', 'kalm-molmo', 'kalm-pixtral'],
                           'vit': ['vit', 'vit-clip', 'vit-dino'],
                           'resnet': ['resnet', 'resnet-dino', 'resnet-clip']}

capitalise_names = {'vit-clip': 'ViT-CLIP', 'resnet-clip': 'ResNet-CLIP', 'gist': 'GIST'}

names_to_paths = {
                'kalm-coco': ROOT / 'results/caption_embeddings/kalm/kalm_coco.pkl',
                'kalm-llava': ROOT / 'results/caption_embeddings/kalm/kalm_llava.pkl',
                'kalm-molmo': ROOT / 'results/caption_embeddings/kalm/kalm_molmo.pkl',
                'kalm-phi': ROOT / 'results/caption_embeddings/kalm/kalm_phi.pkl',
                'kalm-pixtral': ROOT / 'results/caption_embeddings/kalm/kalm_pixtral.pkl',
                'kalm-qwen': ROOT / 'results/caption_embeddings/kalm/kalm_qwen.pkl',
                

                'llama-coco-last' : ROOT / 'results/caption_embeddings/llama3/llama3_human_last.pkl',
                  'llama-llava-last': ROOT / 'results/caption_embeddings/llama3/llama3_llava-ov_last.pkl',
                  'llama-phi-last':  ROOT / 'results/caption_embeddings/llama3/llama3_phi-4_last.pkl',
                  'llama-pixtral-last': ROOT / 'results/caption_embeddings/llama3/llama3_pixtral_last.pkl',
                  'llama-qwen-last': ROOT / 'results/caption_embeddings/llama3/llama3_qwen-vl_last.pkl',
                  'llama-molmo-last': ROOT / 'results/caption_embeddings/llama3/llama3_molmo_last.pkl',

                  'gpt2-coco-last': ROOT / "results/caption_embeddings/gpt2/gpt2_human_last.pkl",
                  'gpt2-phi-last': ROOT / "results/caption_embeddings/gpt2/gpt2_phi-4_last.pkl",
                  'gpt2-pixtral-last': ROOT / "results/caption_embeddings/gpt2/gpt2_pixtral_last.pkl",
                  'gpt2-llava-last': ROOT / "results/caption_embeddings/gpt2/gpt2_llava-ov_last.pkl",
                  'gpt2-molmo-last': ROOT / "results/caption_embeddings/gpt2/gpt2_molmo_last.pkl",
                  'gpt2-qwen-last': ROOT / "results/caption_embeddings/gpt2/gpt2_qwen-vl_last.pkl",

                  'qwen3-coco-last': ROOT / 'results/caption_embeddings/qwen3/qwen3_human_last.pkl',
                  'qwen3-llava-last': ROOT / 'results/caption_embeddings/qwen3/qwen3_llava-ov_last.pkl',
                  'qwen3-phi-last': ROOT / 'results/caption_embeddings/qwen3/qwen3_phi-4_last.pkl',
                  'qwen3-pixtral-last': ROOT / 'results/caption_embeddings/qwen3/qwen3_pixtral_last.pkl',
                  'qwen3-qwen-last': ROOT / 'results/caption_embeddings/qwen3/qwen3_qwen-vl_last.pkl',
                  'qwen3-molmo-last': ROOT / 'results/caption_embeddings/qwen3/qwen3_molmo_last.pkl',
                  
                  'clip-coco-last': ROOT / 'results/caption_embeddings/clip/clip_human_last.pkl',
                  'clip-llava-last': ROOT / 'results/caption_embeddings/clip/clip_llava-ov_last.pkl',
                  'clip-phi-last':  ROOT / 'results/caption_embeddings/clip/clip_phi-4_last.pkl',
                  'clip-pixtral-last': ROOT / 'results/caption_embeddings/clip/clip_pixtral_last.pkl',
                  'clip-qwen-last': ROOT / 'results/caption_embeddings/clip/clip_qwen-vl_last.pkl',
                  'clip-molmo-last': ROOT / 'results/caption_embeddings/clip/clip_molmo_last.pkl',

                  'bert-coco-cls': ROOT / 'results/caption_embeddings/bert/bert_human_cls.pkl',
                  'bert-llava-cls': ROOT / 'results/caption_embeddings/bert/bert_llava-ov_cls.pkl',
                  'bert-phi-cls': ROOT / 'results/caption_embeddings/bert/bert_phi-4_cls.pkl',
                  'bert-pixtral-cls': ROOT / 'results/caption_embeddings/bert/bert_pixtral_cls.pkl',
                  'bert-qwen-cls': ROOT / 'results/caption_embeddings/bert/bert_qwen-vl_cls.pkl',
                  'bert-molmo-cls': ROOT / 'results/caption_embeddings/bert/bert_molmo_cls.pkl',
                  'bert-loc-narr-cls': ROOT / 'results/caption_embeddings/bert/bert_loc-narr_cls.pkl',

                  'vit': ROOT / 'results/image_features/vit/vit_mean.pkl',
                  'vit-clip': ROOT / 'results/image_features/clip/clip_vit_mean.pkl',
                  'vit-dino': ROOT / 'results/image_features/dino/dino_vit.pkl',
                  'resnet': ROOT / 'results/image_features/resnet50/resnet50.pkl',
                  'resnet-dino': ROOT / 'results/image_features/dino/dino_rn50.pkl',
                  'resnet-clip': ROOT / 'results/image_features/clip/clip_rn50.pkl',
                  }

models = list(names_to_paths.keys())

caption_embeddings = ['llama-human-last',
 'llama-human-avg',
 'llama-llava-last',
 'llama-llava-avg',
 'llama-phi-last',
 'llama-phi-avg',
 'llama-pixtral-last',
 'llama-pixtral-avg',
 'llama-qwen-last',
 'llama-qwen-avg',
 'llama-molmo-last',
 'llama-molmo-avg',

 'clip-ce-human-last',
 'clip-ce-human-avg',
 'clip-ce-llava-avg',
 'clip-ce-llava-last',
 'clip-ce-phi-avg',
 'clip-ce-phi-last',
 'clip-ce-pixtral-avg',
 'clip-ce-pixtral-last',
 'clip-ce-qwen-avg',
 'clip-ce-qwen-last',
 'clip-ce-molmo-avg',
 'clip-ce-molmo-last',

 'qwen3-human-last',
 'qwen3-llava-last',
 'qwen3-phi-last',
 'qwen3-pixtral-last',
 'qwen3-qwen-last',
 'qwen3-molmo-last',

 'clip-human-last',
 'clip-human-avg',
 'clip-llava-last',
 'clip-llava-avg',
 'clip-phi-last',
 'clip-phi-avg',
 'clip-pixtral-last',
 'clip-pixtral-avg',
 'clip-qwen-last',
 'clip-qwen-avg',
 'clip-molmo-last',
 'clip-molmo-avg',
 
 'bert-human-cls',
 'bert-human-avg',
 'bert-llava-cls',
 'bert-llava-avg',
 'bert-phi-cls',
 'bert-phi-avg',
 'bert-pixtral-cls',
 'bert-pixtral-avg',
 'bert-qwen-cls',
 'bert-qwen-avg',
 'bert-molmo-cls',
 'bert-molmo-avg'
 ]