import clip 
from clip.clip import _transform
from utils import NSDDataset
from torch.utils.data import DataLoader
import torch
from src.paths import ROOT
import os
from src.utils import save_pickle
from types import MethodType

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def patched_forward(self, x):
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x
    
    def flatten(feat):
    # Global average over spatial dimensions (assumes (B,C,H,W))
        return feat.mean(dim=[2, 3])

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    stem_out = flatten(x)

    x = self.layer1(x)
    layer1_out = flatten(x)

    x = self.layer2(x)
    layer2_out = flatten(x)

    x = self.layer3(x)
    layer3_out = flatten(x)

    x = self.layer4(x)
    layer4_out = flatten(x)

    x = self.attnpool(x)
    attnpool_out = x

    return [stem_out, layer1_out, layer2_out, layer3_out, layer4_out, attnpool_out]


clip_model = clip.load('RN50', device=device, jit=False)
clip_model[0].visual.forward = MethodType(patched_forward, clip_model[0].visual)

clip_resnet = clip_model[0].visual
dataset = NSDDataset(ROOT / 'data/images/stimuli', transform = _transform(224))
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

image_ids = []
img_ftrs = []

clip_resnet.eval()

with torch.no_grad():
    for ids, images in dataloader:
        image_ids += ids
        features = clip_resnet(images.to(device))

        for i in range(len(ids)):
            ftrs = [layer[i].detach().cpu().numpy() for layer in features]
            img_ftrs.append(ftrs)

ftr_dict = {image_id: features for image_id, features in zip(image_ids, img_ftrs)}

(ROOT / f'results/image_features/clip').mkdir(parents=True, exist_ok=True)

save_pickle(ftr_dict, ROOT / f'results/image_features/clip/clip_rn50.pkl')

