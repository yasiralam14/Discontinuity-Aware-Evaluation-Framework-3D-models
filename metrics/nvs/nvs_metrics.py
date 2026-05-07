
from .lpipsPyTorch import lpips
import torch
import cv2
import torchvision.transforms.functional as tf
from tqdm import tqdm
from pathlib import Path
from torch import Tensor
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

def gaussian(window_size: int, sigma: float):
    """ """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()

def ssim(img1, img2, window_size=11, size_average=True, mask: Tensor | None = None):
    channel = img1.size(-3)
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask: Tensor | None = None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if mask is not None:
        assert ssim_map.shape == mask.shape
        ssim_map = ssim_map * mask

    if size_average:
        return ssim_map.sum() / mask.sum().clamp(min=1e-8)
    else:
        return ssim_map.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1e-8)


def psnr(img1, img2, mask: Tensor | None = None):
    # Flatten error to [Batch, Pixels]
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1)

    if mask is not None:
        # Flatten mask to match MSE shape [Batch, Pixels]
        mask_flat = mask.view(mask.shape[0], -1)
        
        # Zero out the MSE in excluded areas
        mse = mse * mask_flat
        
        # Calculate mean using ONLY the count of pixels inside the mask
        # Sum of errors / Number of non-zero mask pixels
        mse = mse.sum(1, keepdim=True) / mask_flat.sum(1, keepdim=True)
    else:
        mse = mse.mean(1, keepdim=True)
        
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def nvs_eval(render: Tensor, gt: Tensor, mask: Tensor | None = None) -> tuple[float, float, float]:
    """
    Computes NVS metrics (PSNR, SSIM, LPIPS) for a single image pair.
    
    :param render: Rendered image tensor of shape (C, H, W).
    :param gt: Ground-truth image tensor of shape (C, H, W).
    :param mask: Optional mask tensor.
    :return: SSIM, PSNR, LPIPS scores.
    """
    # Ensure inputs are (1, C, H, W), select RGB channels, and move to CUDA
    # tf.to_tensor returns (C, H, W), so we unsqueeze to add batch dim
    render = render[:, :3, :, :].cuda()
    gt = gt[:, :3, :, :].cuda()
    mask = mask[:, :3, :, :].cuda()
    
    debug_mask = torch.zeros_like(gt)
    # mask = debug_mask

    # Calculate metrics
    val_ssim = float(ssim(render, gt, mask=mask))
    val_psnr = float(psnr(render, gt, mask=mask.view(mask.shape[0], -1)))
    val_lpips = float(lpips(render, gt, net_type='vgg', mask=mask))

    return val_ssim, val_psnr, val_lpips