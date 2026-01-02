import os
import sys
import logging
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from utils import data_io


class TensorBoardLogger:
    def __init__(self, log_dir: str, loss_type: str, target_type: str):
        self.loss_type = loss_type
        self.target_type = target_type

        self.prefix = {
            'cubic': 'analytical_function',
            'sinusoidal': 'analytical_function',
            'linear': 'analytical_function',
            'image': 'image',
            'helmholtz': 'helmholtz',
            'neuralclothsim': 'neuralclothsim',
            'sdf': 'sdf',
            'advection': 'advection',
            'heat': 'heat',
            'zalesak': 'zalesak',
        }.get(target_type, 'default')

        self.writer = PrefixSummaryWriter(log_dir, self.prefix)

    def log_summary(self, i: int, pred_signal: torch.Tensor, **kwargs: Dict[str, Any]):
        match self.target_type:
            case 'cubic' | 'sinusoidal' | 'linear':
                self.log_analytical_function_summary(pred_signal, kwargs['pred_gradient'], kwargs['pred_laplacian'], kwargs['gt_signal'], kwargs['gt_gradient'], kwargs['gt_laplacian'], i)
            case 'image':
                self.log_image_summary(pred_signal, kwargs['pred_gradient'], kwargs['pred_laplacian'], kwargs['gt_signal'], kwargs['gt_gradient'], kwargs['gt_laplacian'], i)
            case 'helmholtz':
                self.log_helmholtz_summary(pred_signal, kwargs['gt_signal'], i)
            case 'neuralclothsim':
                self.log_neuralclothsim_summary(pred_signal, kwargs['template_mesh'], i)
            case 'sdf':
                self.log_sdf_summary(pred_signal, kwargs['n_test_samples_per_dim'], kwargs['module'], i)
            case 'advection':
                self.log_advection_summary(pred_signal, kwargs['n_test_samples_per_dim'], kwargs['gt_signal'], kwargs['xcoords'], i)
            case 'heat':
                self.log_heat_summary(pred_signal, kwargs['n_test_samples_per_dim'], kwargs['gt_signal'], i)
            case 'zalesak':
                self.log_zalesak_summary(pred_signal, kwargs['n_test_samples_per_dim'], kwargs['gt_signal'], i)

    def log_analytical_function_summary(self, pred_signal: torch.Tensor, pred_gradient: torch.Tensor, pred_laplacian: torch.Tensor, gt_signal: torch.Tensor, gt_gradient: torch.Tensor, gt_laplacian: torch.Tensor, i: int):
        self.writer.add_figure('signal', data_io.get_plot_1d_signal(pred_signal, gt_signal), i)
        # Log as scalar values
        for idx, (pred, gt) in enumerate(zip(pred_signal.view(-1), gt_signal.view(-1))):
            self.writer.add_scalar('signal/prediction', pred.item(), idx)
            self.writer.add_scalar('signal/ground_truth', gt.item(), idx)
            
        if pred_gradient is not None:
            self.writer.add_figure('gradient', data_io.get_plot_1d_signal(pred_gradient[..., 0], gt_gradient), i)
            for idx, (grad_pred, grad_gt) in enumerate(zip(pred_gradient[..., 0].view(-1), gt_gradient.view(-1))):
                self.writer.add_scalar('gradient/prediction', grad_pred.item(), idx)
                self.writer.add_scalar('gradient/ground_truth', grad_gt.item(), idx)

        if pred_laplacian is not None:
            self.writer.add_figure('laplacian', data_io.get_plot_1d_signal(pred_laplacian, gt_laplacian), i)
            for idx, (lap_pred, lap_gt) in enumerate(zip(pred_laplacian[..., 0].view(-1), gt_laplacian.view(-1))):
                self.writer.add_scalar('laplacian/prediction', lap_pred.item(), idx)
                self.writer.add_scalar('laplacian/ground_truth', lap_gt.item(), idx)

    def log_image_summary(self, pred_signal: torch.Tensor, pred_gradient: torch.Tensor, pred_laplacian: torch.Tensor, gt_signal: torch.Tensor, gt_gradient: torch.Tensor, gt_laplacian: torch.Tensor, i: int):
        gt_image = data_io.lin2img(gt_signal)
        pred_image = data_io.lin2img(pred_signal)
        
        gt_gradient_image = data_io.grads2img(gt_gradient)
        pred_gradient_image = data_io.grads2img(pred_gradient)
        
        gt_laplacian_image = data_io.laplace2img(gt_laplacian)

        if self.loss_type in ['gradient', 'laplacian']:
            gt_image = data_io.rescale_img(gt_image)
            pred_image = data_io.rescale_img(pred_image, perc=1e-2)

        self.writer.add_image('signal/gt_vs_pred', make_grid(torch.cat((gt_image, pred_image), dim=-1), normalize=True), i)
        self.writer.add_image('gradient/gt_vs_pred', make_grid(torch.cat((gt_gradient_image, pred_gradient_image), dim=-1), normalize=True), i)
        
        pred_laplacian_image = data_io.laplace2img(pred_laplacian)
        self.writer.add_image('laplacian/gt_vs_pred', make_grid(torch.cat((gt_laplacian_image, pred_laplacian_image), dim=-1), normalize=True), i)

    def log_helmholtz_summary(self, pred_signal: torch.Tensor, gt_signal: torch.Tensor, i: int):
        gt_field = data_io.lin2img(gt_signal)
        gt_field_cmpl = gt_field[...,0,:,:].cpu().numpy() + 1j * gt_field[...,1::2,:,:].cpu().numpy()
        gt_angle = torch.from_numpy(np.angle(gt_field_cmpl))
        gt_mag = torch.from_numpy(np.abs(gt_field_cmpl))

        gt_field = data_io.scale_percentile(gt_field)
        gt_angle = data_io.scale_percentile(gt_angle)
        gt_mag = data_io.scale_percentile(gt_mag)

        pred_signal = data_io.lin2img(pred_signal)
        pred_cmpl = pred_signal[...,0::2,:,:].cpu().numpy() + 1j * pred_signal[...,1::2,:,:].cpu().numpy()
        pred_angle = torch.from_numpy(np.angle(pred_cmpl))
        pred_mag = torch.from_numpy(np.abs(pred_cmpl))

        pred_signal = data_io.scale_percentile(pred_signal)
        pred_angle = data_io.scale_percentile(pred_angle)
        pred_mag = data_io.scale_percentile(pred_mag)

        self.writer.add_image('real/gt_vs_pred', make_grid(torch.cat((data_io.helmholtz2img(gt_field[...,0:1,:,:]), data_io.helmholtz2img(pred_signal[...,0:1,:,:])), dim=-1), normalize=True), i)
        self.writer.add_image('imaginary/gt_vs_pred', make_grid(torch.cat((data_io.helmholtz2img(gt_field[...,1:2,:,:]), data_io.helmholtz2img(pred_signal[...,1:2,:,:])), dim=-1), normalize=True), i)
        self.writer.add_image('angle/gt_vs_pred', make_grid(torch.cat((gt_angle, pred_angle), dim=-1), normalize=True), i)
        self.writer.add_image('magnitude/gt_vs_pred', make_grid(torch.cat((gt_mag, pred_mag), dim=-1), normalize=True), i)

    def log_neuralclothsim_summary(self, pred_signal: torch.Tensor, template_mesh: data_io.Mesh, i: int):
        test_deformed_positions = template_mesh.verts + pred_signal
        # template_mesh.faces values should lie in [0, number_of_vertices] for type `uint8`.
        # self.writer.add_mesh(f'{prefix}/simulated_states', test_deformed_positions, faces=template_mesh.faces.byte(), global_step=i)

        meshes_dir = os.path.join(self.writer.log_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)

        mesh_path = os.path.join(meshes_dir, f'simulated_mesh_step_{i}.obj')
        data_io.save_mesh(mesh_path, test_deformed_positions, template_mesh.faces)

    def log_sdf_summary(self, pred_signal: torch.Tensor, n_test_samples_per_dim: int, module: torch.nn.Module, i: int):
        # Extract slices for contour plotting from pred_signal
        slice_mid = n_test_samples_per_dim // 2

        xy_slice = pred_signal[:, slice_mid::n_test_samples_per_dim, :]
        yz_slice = pred_signal[:, n_test_samples_per_dim ** 2 * (slice_mid - 1):n_test_samples_per_dim ** 2 * slice_mid, :]

        xy_fig = data_io.make_contour_plot(data_io.lin2img(xy_slice).squeeze().cpu().numpy())
        yz_fig = data_io.make_contour_plot(data_io.lin2img(yz_slice).squeeze().cpu().numpy())
        #xz_fig = data_io.make_contour_plot(data_io.lin2img(xz_slice).squeeze().cpu().numpy())

        self.writer.add_figure('sdf/xy_slice', xy_fig, i)
        self.writer.add_figure('sdf/yz_slice', yz_fig, i)
        #self.writer.add_figure('sdf/xz_slice', xz_fig, i)
        meshes_dir = os.path.join(self.writer.log_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)

        mesh_path = os.path.join(meshes_dir, f'extracted_mesh_{i}')
        data_io.create_mesh(module, mesh_path, N=n_test_samples_per_dim)

    def log_advection_summary(self, pred_signal: torch.Tensor, n_test_samples_per_dim: int, gt_signal: torch.Tensor, xcoords: torch.Tensor, i: int):
        # pred_signal: [1, n_test_samples_per_dim ** n_coord_dims, 1]
        # xcoords: [1, n_test_samples_per_dim ** n_spatial_dims, n_spatial_dims]
        n_spatial_dims = xcoords.shape[-1]
        N = n_test_samples_per_dim

        if n_spatial_dims == 1:
            t0_slice = pred_signal[:, :N]
            tmax_slice = pred_signal[:, -N:]
             # Log as images
            self.writer.add_figure(f'advected_field_at_t_0', data_io.get_plot_1d_signal(t0_slice, x=xcoords), i)
            self.writer.add_figure(f'advected_field_at_t_2', data_io.get_plot_1d_signal(tmax_slice, x=xcoords), i)
             # Log as scalar values
            for idx, pred in enumerate(t0_slice.view(-1)):
                self.writer.add_scalar('advected_field_at_t_0', pred.item(), idx)
            for idx, pred in enumerate(tmax_slice.view(-1)):
                self.writer.add_scalar('advected_field_at_t_2', pred.item(), idx)

            gt_signal = gt_signal.view_as(pred_signal)
            gt_t0_slice = gt_signal[:, :N]
            gt_tmax_slice = gt_signal[:, -N:]
            self.writer.add_figure(f'gt_advected_field_at_t_0', data_io.get_plot_1d_signal(gt_t0_slice, x=xcoords), i)
            self.writer.add_figure(f'gt_advected_field_at_t_2', data_io.get_plot_1d_signal(gt_tmax_slice, x=xcoords), i)
        elif n_spatial_dims == 2:
            # t varies fastest, so for each fixed spatial grid, t slices are contiguous
            # Reshape to [N_t, N_x, N_y]
            pred_signal_reshaped = pred_signal.view(1, N, N*N, 1) if pred_signal.shape[1] == N*N*N else pred_signal
            gt_signal_reshaped = gt_signal.view_as(pred_signal_reshaped)
            
            t0_pred = pred_signal_reshaped[0, 0, :, 0]
            tmax_pred = pred_signal_reshaped[0, -1, :, 0]
            t0_gt = gt_signal_reshaped[0, 0, :, 0]
            tmax_gt = gt_signal_reshaped[0, -1, :, 0]
            
            self.writer.add_figure('advected_field_at_t0_pred', data_io.get_plot_single_tensor(t0_pred), i)
            self.writer.add_figure('advected_field_at_tmax_pred', data_io.get_plot_single_tensor(tmax_pred), i)
            self.writer.add_figure('advected_field_at_t0_gt', data_io.get_plot_single_tensor(t0_gt), i)
            self.writer.add_figure('advected_field_at_tmax_gt', data_io.get_plot_single_tensor(tmax_gt), i)

    def log_heat_summary(self, pred_signal: torch.Tensor, n_test_samples_per_dim: int, gt_signal: torch.Tensor, i: int):
        # For 1D: pred_signal shape: [1, N, 1] where N = n_test_samples_per_dim**2 (t,x grid)
        # We'll plot the first and last time slices for both pred and GT
        
        t0_slice = pred_signal[:, :n_test_samples_per_dim]
        tmax_slice = pred_signal[:, -n_test_samples_per_dim:]
        self.writer.add_figure('heat_field_at_t0_pred', data_io.get_plot_1d_signal(t0_slice), i)
        self.writer.add_figure('heat_field_at_tmax_pred', data_io.get_plot_1d_signal(tmax_slice), i)
        for idx, pred in enumerate(t0_slice.view(-1)):
            self.writer.add_scalar('heat_field_at_t0_pred', pred.item(), idx)
        for idx, pred in enumerate(tmax_slice.view(-1)):
            self.writer.add_scalar('heat_field_at_tmax_pred', pred.item(), idx)
        
        gt_t0_slice = gt_signal[:, :n_test_samples_per_dim]
        gt_tmax_slice = gt_signal[:, -n_test_samples_per_dim:]
        self.writer.add_figure('heat_field_at_t0_gt', data_io.get_plot_1d_signal(gt_t0_slice), i)
        self.writer.add_figure('heat_field_at_tmax_gt', data_io.get_plot_1d_signal(gt_tmax_slice), i)
        for idx, gt in enumerate(gt_t0_slice.view(-1)):
            self.writer.add_scalar('heat_field_at_t0_gt', gt.item(), idx)
        for idx, gt in enumerate(gt_tmax_slice.view(-1)):
            self.writer.add_scalar('heat_field_at_tmax_gt', gt.item(), idx)
        
        mae = torch.mean(torch.abs(pred_signal - gt_signal))
        self.writer.add_scalar('metrics/heat_mae', mae.item(), i)

    def log_zalesak_summary(self, pred_signal: torch.Tensor, n_test_samples_per_dim: int, gt_signal: torch.Tensor, i: int):
        # Log t=0 and t=max slices as images
        N = n_test_samples_per_dim
        pred_signal_reshaped = pred_signal.view(1, N, N*N, 1) if pred_signal.shape[1] == N*N*N else pred_signal
        gt_signal_reshaped = gt_signal.view_as(pred_signal_reshaped)
        
        t0_pred = pred_signal_reshaped[0, 0, :, 0]
        tmax_pred = pred_signal_reshaped[0, -1, :, 0]
        t0_gt = gt_signal_reshaped[0, 0, :, 0]
        tmax_gt = gt_signal_reshaped[0, -1, :, 0]
        
        self.writer.add_figure('zalesak_at_t0_pred', data_io.get_plot_single_tensor(t0_pred), i)
        self.writer.add_figure('zalesak_at_tmax_pred', data_io.get_plot_single_tensor(tmax_pred), i)
        self.writer.add_figure('zalesak_at_t0_gt', data_io.get_plot_single_tensor(t0_gt), i)
        self.writer.add_figure('zalesak_at_tmax_gt', data_io.get_plot_single_tensor(tmax_gt), i)

    def close(self):
        self.writer.flush()
        self.writer.close()

class PrefixSummaryWriter(SummaryWriter):
    def __init__(self, log_dir, prefix):
        super().__init__(log_dir)
        self.prefix = prefix

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        super().add_scalar(f'{self.prefix}/{tag}', scalar_value, global_step, *args, **kwargs)

    def add_image(self, tag, img_tensor, global_step=None, *args, **kwargs):
        super().add_image(f'{self.prefix}/{tag}', img_tensor, global_step, *args, **kwargs)

    def add_figure(self, tag, figure, global_step=None, *args, **kwargs):
        super().add_figure(f'{self.prefix}/{tag}', figure, global_step, *args, **kwargs)

    def add_text(self, tag, text_string, global_step=None, *args, **kwargs):
        super().add_text(f'{self.prefix}/{tag}', text_string, global_step, *args, **kwargs)


def get_logger(log_dir: str, expt_name: str) -> logging.Logger:
    logger = logging.getLogger('dinf-grid')
    logger.setLevel(logging.DEBUG)

    stdoutHandler = logging.StreamHandler(stream=sys.stdout)    
    errHandler = logging.FileHandler(os.path.join(log_dir, f'{expt_name}.log'))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
    stdoutHandler.setFormatter(fmt)
    errHandler.setFormatter(fmt)

    logger.addHandler(stdoutHandler)
    logger.addHandler(errHandler)
    return logger