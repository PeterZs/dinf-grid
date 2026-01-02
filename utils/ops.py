from typing import Tuple

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.data_io import rescale_img, scale_percentile


def compl_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out

def compl_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, loss_type: str) -> float:
    if loss_type == 'signal':
        pred = rescale_img(pred)
    elif loss_type in ['gradient', 'laplacian']:
        pred = rescale_img(pred, perc=1e-2)
    gt = rescale_img(gt)
    
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1)
    return psnr

def compute_ssim(pred: torch.Tensor, gt: torch.Tensor, loss_type: str) -> float:
    if loss_type == 'signal':
        pred = rescale_img(pred)
    elif loss_type in ['gradient', 'laplacian']:
        pred = rescale_img(pred, perc=1e-2)
    gt = rescale_img(gt)

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    # SSIM expects images in HxW or HxWxC, data_range=1 for normalized images
    ssim_val = structural_similarity(gt_np[0], pred_np[0], data_range=1, channel_axis=-1 if pred_np.ndim == 3 else None)
    return ssim_val

def gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, d: int = 2) -> torch.Tensor:
    q = -0.5 * ((x - mu) ** 2).sum(dim=1)
    norm_const = 1.0 / torch.sqrt(torch.tensor(sigma ** d * (2 * torch.pi) ** d))
    result = norm_const * torch.exp(q / sigma)
    return result

def count_parameters(module: torch.nn.Module) -> str:
    '''Count the number of trainable parameters in a module and return in a readable format.'''
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if num_params >= 1e6:
        return f'{num_params / 1e6:.2f}m'
    elif num_params >= 1e3:
        return f'{num_params / 1e3:.2f}k'
    return str(num_params)

def compute_helmholtz_error(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Compute the Helmholtz error between the predicted and ground truth solutions.
    The error is calculated as the mean absolute error between the real and imaginary parts.
    '''
    pred = scale_percentile(pred)
    gt = scale_percentile(gt)
    pred = pred * mask
    gt = gt * mask
    real_error = torch.mean(torch.abs(pred[..., 0] - gt[..., 0]))
    imag_error = torch.mean(torch.abs(pred[..., 1] - gt[..., 1]))
    return real_error, imag_error