import math
from typing import Any, Dict, Tuple, Union

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import skimage.data
import trimesh
import numpy as np
import pysdf
from .config import device
from utils.diff_operators import jacobian
from utils.ops import compl_div, compl_mul, gaussian
from neuralclothsim.reference_geometry import ReferenceGeometry
from neuralclothsim.material import LinearMaterial
from neuralclothsim.strain import compute_strain
from utils.data_io import get_plot_single_tensor
Image.MAX_IMAGE_PIXELS = None

class PointCloud(Dataset):
    def __init__(self, pointcloud_path: str, keep_aspect_ratio: bool = True):
        mesh = trimesh.load(pointcloud_path, file_type='ply')
        coords = torch.tensor(mesh.vertices, dtype=torch.float32)

        # Normalize point cloud to fit in bounding box (-1, 1)
        coords -= coords.mean(dim=0, keepdim=True)
        if keep_aspect_ratio:
            coord_max = coords.max()
            coord_min = coords.min()
        else:
            coord_max = coords.max(dim=0, keepdim=True).values
            coord_min = coords.min(dim=0, keepdim=True).values

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.
        self.normals = mesh.vertex_normals # # Needed if loss_type is 'pde'
        self.faces = mesh.faces # Needed if loss_type is 'signal'

    def __len__(self):
        return self.coords.shape[0]

    def get_coords(self, random_indices: torch.Tensor) -> torch.Tensor:
        return self.coords[random_indices]


class ImageDataset(Dataset):
    def __init__(self, image_path: str, loss_type: str):
        image = Image.open(image_path) if image_path else Image.fromarray(skimage.data.camera())
        self.width, self.height = image.size
        transform = Compose([
            Resize((self.height, self.width)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
        n_channels = image_tensor.shape[1]
        # Normalize coordinates to (-1, 1)
        x = torch.linspace(-1, 1, self.width)
        y = torch.linspace(-1, 1, self.height)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [num_pixels, 2]
        pixels = image_tensor.permute(0, 2, 3, 1).reshape(1, -1, n_channels)[0]  # [num_pixels, n_channels]
        self.coords = coords
        self.pixels = pixels

        # Gradient and Laplacian computation
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1.]]]])
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1.]]]])
        laplacian_kernel = torch.tensor([[[ [0, 1, 0], [1, -4, 1], [0, 1, 0.]]]])
        # Loss scaling
        image_for_ops = image_tensor.clone()
        if loss_type == 'gradient':
            image_for_ops = image_for_ops * 1e1
        elif loss_type == 'laplacian':
            image_for_ops = image_for_ops * 1e4
        image_for_ops = torch.nn.functional.pad(image_for_ops, (1, 1, 1, 1), mode='reflect')
        gradx = torch.nn.functional.conv2d(image_for_ops, weight=sobel_x_kernel.repeat(n_channels, 1, 1, 1), groups=n_channels)
        grady = torch.nn.functional.conv2d(image_for_ops, weight=sobel_y_kernel.repeat(n_channels, 1, 1, 1), groups=n_channels)
        laplace = torch.nn.functional.conv2d(image_for_ops, weight=laplacian_kernel.repeat(n_channels, 1, 1, 1), groups=n_channels)
        self.signal = image_for_ops[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1).reshape(1, -1, n_channels)  # [1, height * width, channels]
        self.gradient = torch.stack((grady, gradx), dim=-1).permute(0, 2, 3, 1, 4).reshape(1, -1, n_channels, 2)  # [1, height * width, channels, 2]
        self.laplacian = laplace.permute(0, 2, 3, 1).reshape(1, -1, n_channels)  # [1, height * width, channels]

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.pixels[idx]

    def get_coords(self, random_indices: torch.Tensor) -> torch.Tensor:
        return self.coords[random_indices]

    def get_pixels(self, random_indices: torch.Tensor) -> torch.Tensor:
        return self.pixels[random_indices]
    
    def get_gradients(self, random_indices: torch.Tensor) -> torch.Tensor:
        return self.gradient[0, random_indices, :, :]

    def get_laplacians(self, random_indices: torch.Tensor) -> torch.Tensor:
        return self.laplacian[0, random_indices, :]

class Target:
    def __init__(self, args: Any):
        self.target_type = args.target_type
        self.loss_type = args.loss_type
        self.dataset = None
       
        match self.target_type:
            case 'image':
                self.dataset = ImageDataset(image_path=args.image_path, loss_type=args.loss_type)
                args.n_test_samples_per_dim = self.dataset.height # Currently, assuming square images
                self.patch_size = args.patch_size
                self.use_patch = args.use_patch
            case 'helmholtz':
                self.source = torch.tensor([1.0, 1.0]).view(-1, 2)
                self.source_coords = torch.tensor([0., 0.]).view(-1, 2)
                self.sigma = 1e-4
                self.wavenumber = args.wavenumber
            case 'neuralclothsim':
                self.ref_geometry = ReferenceGeometry()
                self.material = LinearMaterial(self.ref_geometry, mass_area_density=0.144, thickness=0.0012, youngs_modulus=5000, poissons_ratio=0.25)
                self.external_load = torch.tensor([0, -9.8, 0]).expand(1, args.n_train_stratified_samples_per_dim** 2, 3) * self.material.mass_area_density
            case 'advection':
                self.mu = torch.tensor(args.advection_mu).view(1, 1, -1)
                self.sigma = torch.tensor(args.advection_sigma).view(1, 1, -1)
                self.velocity = torch.tensor(args.advection_velocity).view(1, 1, -1)
            case 'sdf':
                # bun_zipper.ply, Armadillo.ply, dragon_vrip.ply
                self.dataset = PointCloud(pointcloud_path=args.point_cloud_path)
                
                # Overfitting to SDF of the mesh (i.e., loss_type is signal) 
                self.sdf_fn = pysdf.SDF(self.dataset.coords.cpu().numpy(), self.dataset.faces) 
                
                # Recovering SDF from oriented point cloud (i.e., loss_type is pde)
                on_surface_samples = args.n_train_data_samples_per_dim ** 3
                off_surface_samples = args.n_train_stratified_samples_per_dim ** 3
                self.gt_sdf = torch.zeros(on_surface_samples + off_surface_samples, 1)  # on-surface = 0
                self.gt_sdf[on_surface_samples:, :] = -1  # off-surface = -1
                self.off_surface_normals = torch.full((off_surface_samples, 3), -1.0)
                self.gt_normals = torch.tensor(self.dataset.normals)
            case 'heat':
                self.alpha = 1.0
            case 'zalesak':
                # Parameters for slotted disk
                self.radius = args.zalesak_radius
                self.slot_width = args.zalesak_slot_width
                self.slot_height = args.zalesak_slot_height
                self.disk_center_x = args.zalesak_center_x
                self.disk_center_y = args.zalesak_center_y
                self.angular_velocity = math.pi/2  # rigid rotation
                
                self.rotation_center_x = 0.
                self.rotation_center_y = 0.

                self.slot_xmin = self.disk_center_x - self.slot_width / 2
                self.slot_xmax = self.disk_center_x + self.slot_width / 2
                self.slot_ymin = self.disk_center_y
                self.slot_ymax = self.disk_center_y + self.slot_height

    def compute(
        self,
        coords: torch.Tensor,
        pred_signal: Union[torch.Tensor, None] = None,
        **kwargs: Dict[str, Any],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        match self.target_type:
            case 'linear':
                return self._linear(coords)
            case 'cubic':
                return self._cubic(coords)
            case 'sinusoidal':
                return self._sinusoidal(coords)
            case 'image':
                random_indices = kwargs['random_indices']
                return self.dataset.get_pixels(random_indices), self.dataset.get_gradients(random_indices), self.dataset.get_laplacians(random_indices)
            case 'helmholtz':
                return self._helmholtz(pred_signal, coords)
            case 'neuralclothsim':
                return self._neuralclothsim(pred_signal, coords)
            case 'sdf':
                if self.loss_type == 'pde':
                    return self._eikonal(pred_signal, coords, kwargs['random_indices'])
                else:
                    return self._get_sdf(coords)
            case 'advection':
                return self._advection(pred_signal, coords)
            case 'heat':
                return self._heat(pred_signal, coords)
            case 'zalesak':
                return self._zalesak_pde(pred_signal, coords)

    def _linear(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal = coords.detach()
        gradient = torch.ones_like(signal[...,None])
        laplacian = torch.zeros_like(signal)
        return signal, gradient, laplacian

    def _cubic(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal = coords.detach() ** 3 # [1, n_samples, 1]
        gradient = 3 * coords[...,None].detach() ** 2 # [1, n_samples, 1, 1]
        laplacian = 6 * coords.detach() # [1, n_samples, 1]
        return signal, gradient, laplacian

    def _sinusoidal(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal = torch.sin(coords.detach())
        gradient = torch.cos(coords[...,None].detach())
        laplacian = -torch.sin(coords.detach())
        return signal, gradient, laplacian

    def _helmholtz(self, pred_signal: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        mask = (coords >= -0.05) & (coords <= 0.05)
        mask = mask.all(dim=2)
        assert mask.sum() >= 100, f'Not enough source coords (coords in the range [-0.05, 0.05]) to sample from. Found {mask.sum()}.'
        
        # We use the value "zero" to encode "no boundary constraint at this coordinate"
        source_boundary_values = self.source * gaussian(coords[0].detach(), mu=self.source_coords, sigma=self.sigma)[:, None]
        source_boundary_values[source_boundary_values < 1e-5] = 0.
        source_boundary_values = source_boundary_values[None]
        
        x = coords  # [1, num_points, 2]
        y = pred_signal # [1, num_points, 2]

        # specify squared slowness
        squared_slowness = torch.ones_like(coords[0].detach())
        squared_slowness[..., 1] = 0. 
        squared_slowness = squared_slowness.repeat(1, 1, y.shape[-1] // 2)
        num_samples = x.shape[1]

        du = jacobian(y, x)
        dudx1 = du[..., 0]
        dudx2 = du[..., 1]

        a0 = 5.0

        # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
        Lpml = 0.5
        dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
        dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
        dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
        dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

        sx = self.wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
        sy = self.wavenumber * a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

        ex = torch.cat((torch.ones_like(sx), -sx / self.wavenumber), dim=-1)
        ey = torch.cat((torch.ones_like(sy), -sy / self.wavenumber), dim=-1)

        A = compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
        B = compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
        C = compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

        a = jacobian(compl_mul(A, dudx1), x)
        b = jacobian(compl_mul(B, dudx2), x)

        a = a[..., 0]
        b = b[..., 1]
        c = compl_mul(compl_mul(C, squared_slowness), self.wavenumber ** 2 * y)

        diff_constraint_hom = a + b + c
        diff_constraint_on = torch.where(source_boundary_values != 0., diff_constraint_hom - source_boundary_values, torch.zeros_like(diff_constraint_hom))
        diff_constraint_off = torch.where(source_boundary_values == 0., diff_constraint_hom, torch.zeros_like(diff_constraint_hom))

        loss_diff_constraint_on = torch.abs(diff_constraint_on).sum() * num_samples / 1e3
        loss_diff_constraint_off = torch.abs(diff_constraint_off).sum()
        return loss_diff_constraint_on + loss_diff_constraint_off

    def _neuralclothsim(self, pred_deformation: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        self.ref_geometry(coords)
        strain = compute_strain(pred_deformation, self.ref_geometry)
         
        hyperelastic_strain_energy = self.material.compute_internal_energy(strain)
        external_energy = torch.einsum('ijk,ijk->ij', self.external_load, pred_deformation)
        mechanical_energy = (hyperelastic_strain_energy - external_energy) * torch.sqrt(self.ref_geometry.a)
        '''fig = get_plot_single_tensor(hyperelastic_strain_energy)
        fig.savefig(f'logs/hyperelastic_strain_energy.png')'''
        return mechanical_energy.mean()

    def _eikonal(self, pred_sdf: torch.Tensor, coords: torch.Tensor, random_indices: torch.Tensor) -> torch.Tensor:
        on_surface_normals = self.gt_normals[random_indices.squeeze()]
        gt_normals = torch.cat((on_surface_normals, self.off_surface_normals), dim=0)

        grad = jacobian(pred_sdf, coords)[..., 0, :]

        # Constraints
        eikonal_constraint = torch.abs(grad.norm(dim=-1) - 1)
        
        sdf_constraint = torch.where(self.gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))        
        normal_constraint = torch.where(self.gt_sdf != -1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None], torch.zeros_like(grad[..., :1]))
        
        inter_constraint = torch.where(self.gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))

        # Weighted loss components
        losses = {'sdf': torch.abs(sdf_constraint).mean() * 3e3,
            'inter': inter_constraint.mean() * 1e2,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'eikonal_constraint': eikonal_constraint.mean() * 5e1
        }
        return sum(losses.values())

    def _get_sdf(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_samples, n_dims = coords.shape
        coords_flat = coords.reshape(-1, n_dims).detach().cpu().numpy()
        
        sdf_np = np.asarray(self.sdf_fn(coords_flat), dtype=np.float32).reshape(-1, 1)
        sdf = torch.from_numpy(sdf_np).to(device=coords.device)
        sdf = sdf.view(batch_size, num_samples, 1)
        
        zeros_grad = torch.zeros(batch_size, num_samples, 1, n_dims)
        zeros_laplacian = torch.zeros(batch_size, num_samples, 1)
        return sdf, zeros_grad, zeros_laplacian

    def _advection(self, pred_advected_quantity: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        gradient = jacobian(pred_advected_quantity, coords)
        du_dt = gradient[..., 0]
        du_dx = gradient[..., 1:]
        # Compute the advection PDE residual
        residual = du_dt + (self.velocity * du_dx).sum(dim=-1)
        return torch.abs(residual).mean()

    def _heat(self, pred_quantity: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        grad = jacobian(pred_quantity, coords)
        du_dt = grad[..., 0]
        laplacian = 0
        for d in range(1, coords.shape[-1]):
            # Take grad wrt spatial dim d
            grad_spatial = jacobian(grad[..., d], coords)[..., d]
            laplacian = laplacian + grad_spatial
        residual = du_dt - self.alpha * laplacian
        return torch.abs(residual).mean()

    def _zalesak_pde(self, pred_advected_quantity: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # Advection PDE for Zalesak: velocity is rigid rotation
        gradient = jacobian(pred_advected_quantity, coords)
        du_dt = gradient[..., 0]
        du_dx = gradient[..., 1:]
        
        velocity = torch.zeros_like(coords[..., None, 1:])
        velocity[..., 0] = -(coords[..., 2:3].detach() - self.rotation_center_y) * self.angular_velocity
        velocity[..., 1] = (coords[..., 1:2].detach() - self.rotation_center_x) * self.angular_velocity

        residual = du_dt + (velocity * du_dx).sum(dim=-1)
        return torch.abs(residual).mean()
