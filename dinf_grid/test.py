import math
from typing import Any, Dict, Tuple, Union

import torch
import numpy as np
import scipy.special

from utils.ops import compute_psnr, compute_ssim, compute_helmholtz_error
from utils.diff_operators import jacobian, laplacian
from utils.data_io import generate_mesh_topology, lin2img, Mesh
from .sampler import get_mgrid
from .config import device
from neuralclothsim.reference_geometry import midsurface
from .logger import TensorBoardLogger
from .target import Target

class Test:
    def __init__(
        self,
        module: torch.nn.Module,
        target: Target,
        n_test_samples_per_dim: int,
        n_coord_dims: int,
        coord_space: Tuple[float, ...],
        tb_logger: TensorBoardLogger,
    ):
        self.module = module
        self.n_test_samples_per_dim = n_test_samples_per_dim
        self.tb_logger = tb_logger
        self.test_coords = get_mgrid(n_coord_dims * (self.n_test_samples_per_dim,), stratified=False, dim=n_coord_dims)[None]
        for i in range(n_coord_dims):
            self.test_coords[..., i] *= coord_space[i]
        match self.tb_logger.target_type:
            case 'cubic' | 'sinusoidal' | 'linear':
                self.gt_signal, self.gt_gradient, self.gt_laplacian = target.compute(self.test_coords)
                self.channel = 0
            case 'image':
                self.gt_signal = target.dataset.signal
                self.gt_gradient = target.dataset.gradient
                self.gt_laplacian = target.dataset.laplacian
                self.channel = 0 # red
                self.use_patch = target.use_patch
                self.patch_size = target.patch_size
            case 'neuralclothsim':
                faces = torch.tensor(generate_mesh_topology(self.n_test_samples_per_dim)[None])
                self.template_mesh = Mesh(verts=midsurface(self.test_coords), faces=faces, curvilinear_coords=self.test_coords)
            case 'helmholtz':
                # For reference: this derives the closed-form solution for the inhomogenous Helmholtz equation.
                square_meshgrid = lin2img(self.test_coords).cpu().numpy()
                x = square_meshgrid[0, 0, ...]
                y = square_meshgrid[0, 1, ...]
                
                # Specify the source.
                source_np = target.source.cpu().numpy()
                hx = hy = 2 / n_test_samples_per_dim
                field = np.zeros((n_test_samples_per_dim, n_test_samples_per_dim)).astype(np.complex64)
                for i in range(source_np.shape[0]):
                    x0 = target.source_coords[i, 0].cpu().numpy()
                    y0 = target.source_coords[i, 1].cpu().numpy()
                    s = source_np[i, 0] + 1j * source_np[i, 1]

                    hankel = scipy.special.hankel2(0, target.wavenumber * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 1e-6)
                    field += 0.25j * hankel * s * hx * hy

                field_r = torch.from_numpy(np.real(field).reshape(-1, 1))
                field_i = torch.from_numpy(np.imag(field).reshape(-1, 1))
                self.field = torch.cat((field_r, field_i), dim=1)[None].to(device)
                
                # Consider only the interior of the domain [-0.5, 0.5] ** 2 for metrics.
                mask = (self.test_coords >= -0.5) & (self.test_coords <= 0.5)
                self.mask = mask.all(dim=-1, keepdim=True)
            case 'advection':
                t_grid = self.test_coords[..., 0] + coord_space[0]
                x_grid = self.test_coords[..., 1:]
                # u(x,t) = u_0(x), where u_0(x) = exp(-(x - mu)^2 / (2 * sigma^2))
                # u(x,t) = exp(-(x - (mu + velocity * (t - t_0)))^2 / (2 * sigma^2))
                
                # Support both 1D and 2D advection
                # For each spatial dim x_i: mu_i + velocity_i * t
                adv_center = target.mu + target.velocity * t_grid.unsqueeze(-1)
                # Multivariate Gaussian: exp(-sum_i ((x_i - center_i)^2 / (2*sigma_i^2)))
                exponent = ((x_grid - adv_center) ** 2) / (2 * target.sigma ** 2)
                self.gt_signal = torch.exp(-exponent.sum(dim=-1)).squeeze()
                self.x_initial = x_grid[0, :self.n_test_samples_per_dim * (n_coord_dims - 1)].clone()
            case 'heat':
                # Sinusoidal initial condition: u(x,0) = sin(pi*x)
                # Prepare GT closed-form solution for heat equation
                # Closed-form solution for u(x, t) = sin(pi x) * exp(-alpha * pi^2 * t)
                t_grid = self.test_coords[..., 0] + coord_space[0]
                x_grid = self.test_coords[..., 1]
                self.gt_signal = (torch.sin(math.pi * x_grid) * torch.exp(-target.alpha * math.pi ** 2 * t_grid))[..., None]
            case 'zalesak':
                # GT: slotted disk SDF advected by rigid rotation
                t = self.test_coords[..., 0] + coord_space[0]
                x = self.test_coords[..., 1]
                y = self.test_coords[..., 2]
                theta = -target.angular_velocity * t

                # Backtrace to initial time
                x_rot = (x - target.rotation_center_x) * torch.cos(theta) - (y - target.rotation_center_y) * torch.sin(theta) + target.rotation_center_x
                y_rot = (x - target.rotation_center_x) * torch.sin(theta) + (y - target.rotation_center_y) * torch.cos(theta) + target.rotation_center_y

                # SDF for disk
                sdf_disk = torch.sqrt((x_rot - target.disk_center_x) ** 2 + (y_rot - target.disk_center_y) ** 2) - target.radius

                # SDF for slot (rectangle)
                dx = torch.max(torch.stack([target.slot_xmin - x_rot, x_rot - target.slot_xmax, torch.zeros_like(x_rot)]), dim=0).values
                dy = torch.max(torch.stack([target.slot_ymin - y_rot, y_rot - target.slot_ymax, torch.zeros_like(y_rot)]), dim=0).values
                sdf_slot = torch.sqrt(dx ** 2 + dy ** 2)
                
                inside_slot = (target.slot_xmin <= x_rot) & (x_rot <= target.slot_xmax) & (target.slot_ymin <= y_rot) & (y_rot <= target.slot_ymax)
                sdf_slot[inside_slot] = -torch.min(
                    torch.stack([
                        x_rot[inside_slot] - target.slot_xmin,
                        target.slot_xmax - x_rot[inside_slot],
                        y_rot[inside_slot] - target.slot_ymin,
                        target.slot_ymax - y_rot[inside_slot]
                    ]),
                    dim=0
                ).values
                
                # SDF for slotted disk
                self.gt_signal = torch.max(sdf_disk, -sdf_slot)[..., None]

    def __call__(self, i: int) -> None:
        if self.tb_logger.target_type in ['cubic', 'sinusoidal', 'linear', 'image']:
            if self.tb_logger.target_type == 'image' and self.use_patch:
                # Random patch location
                top = np.random.randint(0, self.n_test_samples_per_dim - self.patch_size)
                left = np.random.randint(0, self.n_test_samples_per_dim - self.patch_size)
                patch_indices = []
                for y in range(top, top + self.patch_size):
                    for x in range(left, left + self.patch_size):
                        patch_indices.append(y * self.n_test_samples_per_dim + x)
                patch_indices = torch.tensor(patch_indices, device=device)
                test_coords = self.test_coords[:, patch_indices, :].requires_grad_(True)
                gt_signal = self.gt_signal[:, patch_indices, :]
                gt_gradient = self.gt_gradient[:, patch_indices, :, :]
                gt_laplacian = self.gt_laplacian[:, patch_indices, :]
            else:
                patch_indices = None
                test_coords = self.test_coords.requires_grad_(True)
                gt_signal, gt_gradient, gt_laplacian = self.gt_signal, self.gt_gradient, self.gt_laplacian
            
            pred_signal = self.module(test_coords, train=False, patch_indices=patch_indices)
            pred_gradient = jacobian(pred_signal, test_coords)
            pred_laplacian = laplacian(pred_gradient, test_coords)
            self.tb_logger.log_summary(i, pred_signal, pred_gradient=pred_gradient[...,self.channel,:], pred_laplacian=pred_laplacian, gt_signal=gt_signal, gt_gradient=gt_gradient[...,self.channel,:], gt_laplacian=gt_laplacian)
        
        with torch.no_grad():
            test_pred_signal = self.module(self.test_coords, train=False)
            match self.tb_logger.target_type:
                case 'image':
                    psnr = compute_psnr(test_pred_signal, self.gt_signal, self.tb_logger.loss_type)
                    ssim = compute_ssim(test_pred_signal, self.gt_signal, self.tb_logger.loss_type)
                    self.tb_logger.writer.add_scalar('metrics/psnr', psnr, i)
                    self.tb_logger.writer.add_scalar('metrics/ssim', ssim, i)                    
                case 'neuralclothsim':
                    self.tb_logger.log_summary(i, test_pred_signal, template_mesh=self.template_mesh)
                case 'helmholtz':
                    self.tb_logger.log_summary(i, test_pred_signal, gt_signal=self.field)
                    real_error, imag_error = compute_helmholtz_error(test_pred_signal, self.field, self.mask)
                    self.tb_logger.writer.add_scalar('metrics/real_error', real_error, i)
                    self.tb_logger.writer.add_scalar('metrics/imag_error', imag_error, i)
                case 'advection':
                    mae = torch.mean(torch.abs(test_pred_signal.squeeze() - self.gt_signal))
                    self.tb_logger.writer.add_scalar('metrics/advection_mae', mae, i)
                    self.tb_logger.log_summary(i, test_pred_signal, n_test_samples_per_dim=self.n_test_samples_per_dim, gt_signal=self.gt_signal, xcoords=self.x_initial)
                case 'sdf':
                    self.tb_logger.log_summary(i, test_pred_signal, n_test_samples_per_dim=self.n_test_samples_per_dim, module=self.module)
                case 'heat':
                    self.tb_logger.log_summary(i, test_pred_signal, n_test_samples_per_dim=self.n_test_samples_per_dim, gt_signal=self.gt_signal)
                case 'zalesak':
                    l1_error = torch.mean(torch.abs(test_pred_signal - self.gt_signal))
                    l2_error = torch.sqrt(torch.mean((test_pred_signal - self.gt_signal) ** 2))
                    self.tb_logger.writer.add_scalar('metrics/zalesak_l1_error', l1_error.item(), i)
                    self.tb_logger.writer.add_scalar('metrics/zalesak_l2_error', l2_error.item(), i)
                    self.tb_logger.log_summary(i, test_pred_signal, n_test_samples_per_dim=self.n_test_samples_per_dim, gt_signal=self.gt_signal)
