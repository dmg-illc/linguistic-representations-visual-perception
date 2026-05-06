import torch
from torchvision.models import resnet50
from resnet_patched_fwd import _patched_forward_impl
from torch import Tensor
import pickle
import argparse
from src.paths import ROOT
from src.utils import *
import numpy as np
from my_utils import NSDDataset
from torch.utils.data import DataLoader
from types import MethodType


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50').to(device)

# replacing the original fwd method to obtain all outputs
model._forward_impl = MethodType(_patched_forward_impl, model)

# transforms found here https://github.com/facebookresearch/dino/blob/main/eval_linear.py
current_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

dataset = NSDDataset(ROOT / 'data/images/stimuli', transform=current_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

image_ids = []
img_ftrs = []

model.eval()

with torch.no_grad():
    for ids, images in dataloader:
        image_ids += ids
        features = model(images.to(device))

        for i in range(len(ids)):
            ftrs = [layer[i].detach().cpu().numpy() for layer in features]
            img_ftrs.append(ftrs)

ftr_dict = {image_id: features for image_id, features in zip(image_ids, img_ftrs)}

(ROOT / f'results/image_features/dino').mkdir(parents=True, exist_ok=True)

save_pickle(ftr_dict, ROOT / f'results/image_features/dino/dino_rn50.pkl')