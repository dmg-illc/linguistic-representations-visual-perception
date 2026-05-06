import torch
from torchvision.models import resnet50, ResNet50_Weights
from resnet_patched_fwd import _patched_forward_impl
from src.paths import ROOT
from src.utils import *
import numpy as np
from my_utils import NSDDataset
from torch.utils.data import DataLoader
from types import MethodType

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = ResNet50_Weights.IMAGENET1K_V2  # or IMAGENET1K_V2
model = resnet50(weights=weights).to(device)

# replacing the original fwd method to obtain all outputs
model._forward_impl = MethodType(_patched_forward_impl, model)

dataset = NSDDataset(ROOT / 'data/images/stimuli', transform = weights.transforms())
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

(ROOT / f'results/image_features/resnet50').mkdir(parents=True, exist_ok=True)

save_pickle(ftr_dict, ROOT / f'results/image_features/resnet50/resnet50.pkl')