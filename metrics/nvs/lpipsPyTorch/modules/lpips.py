import torch
import torch.nn as nn
from .networks import get_network, LinLayers
from .utils import get_state_dict
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys




class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
        
    """
    printed_once = False
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        self.net = get_network(net_type)

        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))
        

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None):
        feat_x, feat_y = self.net(x), self.net(y)
        
        if mask is not None and mask.shape[1] > 1:
            mask = mask[:, 0:1, :, :]
        
        res = []
        for fx, fy, l in zip(feat_x, feat_y, self.lin):
            fx = fx / (torch.sqrt(torch.sum(fx**2, dim=1, keepdim=True)) + 1e-10)
            fy = fy / (torch.sqrt(torch.sum(fy**2, dim=1, keepdim=True)) + 1e-10)
            
            diff = (fx - fy) ** 2
            
            dist_map = l(diff)
            
            if mask is not None:

                target_h, target_w = dist_map.shape[-2:]
                
                mask_cpu = mask.detach().cpu()
                resized_masks = []

                for i in range(mask_cpu.shape[0]):
                    m_np = mask_cpu[i, 0].numpy()
                    
                    img_pil = Image.fromarray(m_np)
                    
                    img_resized = img_pil.resize((target_w, target_h), resample=Image.BILINEAR)
                    
                    resized_masks.append(torch.from_numpy(np.array(img_resized)))

                cur_mask = torch.stack(resized_masks).unsqueeze(1).to(dist_map.device)
                cur_mask = cur_mask >=0.5
                
                
                num = (dist_map * cur_mask).sum(dim=(2, 3), keepdim=True)
                den = cur_mask.sum(dim=(2, 3), keepdim=True) + 1e-8
                res.append(num / den)
            else:
                res.append(dist_map.mean(dim=(2, 3), keepdim=True))
                
        return torch.cat(res, 1).sum(dim=1).mean()