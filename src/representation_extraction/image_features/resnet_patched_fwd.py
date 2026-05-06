import torch
from torch import Tensor

def _patched_forward_impl(self, x: Tensor) -> Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    stem_out = x.mean(dim=[2, 3]) # flattening output to save

    x = self.layer1(x)
    layer_1_out = x.mean(dim=[2, 3])
    
    x = self.layer2(x)
    layer_2_out = x.mean(dim=[2, 3])

    x = self.layer3(x)
    layer_3_out = x.mean(dim=[2, 3])

    x = self.layer4(x)
    layer_4_out = x.mean(dim=[2, 3])

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    fc_out = x

    return [stem_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, fc_out]