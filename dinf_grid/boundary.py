import torch
from typing import Tuple
from .target import Target

class Boundary:
    def __init__(self, boundary_condition: str, coord_space: Tuple[float, ...], target: Target):
        self.boundary_condition = boundary_condition
        self.boundary_support = 0.01
        self.coord_space = coord_space
        self.target = target
        
    def normalization(self, coords: torch.Tensor) -> torch.Tensor:        
        normalized_coords = coords.clone()
        for i in range(coords.shape[-1]):
            normalized_coords[..., i] = coords[..., i] / self.coord_space[i]
        return normalized_coords
    
    def dirichlet_condition(self, pred_signal: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        match self.boundary_condition:
            case 'no_boundary':
                pred_signal_with_boundary = pred_signal
            case 'neuralclothsim_origin_fixed':
                distance_weight = torch.exp(-torch.sum(coords ** 2, dim=-1, keepdim=True) / self.boundary_support)
                pred_signal_with_boundary = pred_signal * (1 - distance_weight)
            case 'neuralclothsim_top_left_fixed':
                distance_weight = torch.exp(-torch.sum((coords - 1) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                pred_signal_with_boundary = pred_signal * (1 - distance_weight)
            case 'neuralclothsim_top_left_top_right_moved':
                top_left_corner = torch.exp(-((coords[..., 0:1] + 1) ** 2 + (coords[..., 1:2] - 1) ** 2) / self.boundary_support)
                top_right_corner = torch.exp(-((coords[..., 0:1] - 1) ** 2 + (coords[..., 1:2] - 1) ** 2) / self.boundary_support)
                motion = 0.4 * torch.ones_like(coords[..., 0:1])
                corner_displacement = torch.cat([motion, torch.zeros_like(motion), torch.zeros_like(motion)], dim=2)
                pred_signal_with_boundary = pred_signal * (1 - top_left_corner) * (1 - top_right_corner) + corner_displacement * top_left_corner - corner_displacement * top_right_corner
            case 'advection_boundary':
                temporal_coords = coords[..., 0:1]
                spatial_coords = coords[..., 1:]
                # Handle both 1D and ND (e.g., 2D) spatial coordinates
                # Build boundary masks for each spatial dimension
                boundary_masks = []
                for d in range(spatial_coords.shape[-1]):
                    x_left = torch.exp(-torch.sum((spatial_coords[..., d:d+1] + self.coord_space[d+1]) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                    x_right = torch.exp(-torch.sum((spatial_coords[..., d:d+1] - self.coord_space[d+1]) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                    boundary_masks.append(1 - x_left)
                    boundary_masks.append(1 - x_right)
                # Combine all spatial boundary masks (product: zero at any edge)
                spatial_boundary_mask = torch.ones_like(pred_signal)
                for mask in boundary_masks:
                    spatial_boundary_mask = spatial_boundary_mask * mask

                t0 = torch.exp(-torch.sum((temporal_coords + self.coord_space[0]) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                # Initial condition: multivariate Gaussian                
                exponent = ((spatial_coords - self.target.mu) ** 2) / (2 * self.target.sigma ** 2)
                initial_condition = torch.exp(-exponent.sum(dim=-1, keepdim=True))
                pred_signal_with_boundary = pred_signal * spatial_boundary_mask * (1 - t0) + t0 * initial_condition
            case 'heat_boundary':
                # For heat equation with sinusoidal initial condition: enforce u(x,0) = sin(pi*x), u(-1,t)=u(1,t)=0
                temporal_coords = coords[..., 0:1]
                spatial_coords = coords[..., 1:2]
                t0 = torch.exp(-torch.sum((temporal_coords + self.coord_space[0]) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                x_left = torch.exp(-torch.sum((spatial_coords + self.coord_space[1]) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                x_right = torch.exp(-torch.sum((spatial_coords - self.coord_space[1]) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                # Initial condition at t=0: sin(pi*x)
                initial_condition = torch.sin(torch.pi * spatial_coords)
                pred_signal_with_boundary = pred_signal * (1 - t0) * (1 - x_left) * (1 - x_right) + t0 * initial_condition
            case 'zalesak_boundary':
                # Zalesak's slotted disk initial condition (SDF), slot is rectangle
                # Apply initial condition at t=0, zero elsewhere
                temporal_coords = coords[..., 0:1]
                spatial_coords = coords[..., 1:]
                t0 = torch.exp(-torch.sum((temporal_coords + self.coord_space[0]) ** 2, dim=-1, keepdim=True) / self.boundary_support)
                
                x = spatial_coords[..., 0]
                y = spatial_coords[..., 1]

                sdf_disk = torch.sqrt((x - self.target.disk_center_x) ** 2 + (y - self.target.disk_center_y) ** 2) - self.target.radius

                dx = torch.max(torch.stack([self.target.slot_xmin - x, x - self.target.slot_xmax, torch.zeros_like(x)]), dim=0).values
                dy = torch.max(torch.stack([self.target.slot_ymin - y, y - self.target.slot_ymax, torch.zeros_like(y)]), dim=0).values
                sdf_slot = torch.sqrt(dx ** 2 + dy ** 2)
                inside_slot = (self.target.slot_xmin <= x) & (x <= self.target.slot_xmax) & (self.target.slot_ymin <= y) & (y <= self.target.slot_ymax)
                sdf_slot[inside_slot] = -torch.min(
                    torch.stack([
                        x[inside_slot] - self.target.slot_xmin,
                        self.target.slot_xmax - x[inside_slot],
                        y[inside_slot] - self.target.slot_ymin,
                        self.target.slot_ymax - y[inside_slot]
                    ]),
                    dim=0
                ).values

                # SDF for slotted disk: max(sdf_disk, -sdf_slot)
                initial_condition = torch.max(sdf_disk, -sdf_slot)[..., None]
                pred_signal_with_boundary = pred_signal * (1 - t0) + t0 * initial_condition.detach()
            case _:
                raise ValueError(f'Unknown boundary condition: {self.boundary_condition}')
        return pred_signal_with_boundary
