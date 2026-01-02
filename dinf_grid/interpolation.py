import torch
from torch.utils.data import Dataset
from typing import Any, Dict


class Interpolation:
    def __init__(self, feature_grid: torch.Tensor):
        self.feature_grid = feature_grid

    def interpolate(self, coords: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError('Subclasses must implement this method.')

class LinearInterpolation(Interpolation):
    def __init__(self, feature_grid: torch.Tensor):
        super().__init__(feature_grid)
        self.grid_shape = feature_grid.shape[:-1]

    # Non-recursive n-dimensional linear interpolation
    def interpolate(self, coords: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        n_dims = len(self.grid_shape)
        coord_components = [coords[..., i] for i in range(n_dims)]
        floor_indices = [comp.floor().long() for comp in coord_components]
        ceil_indices = [comp.ceil().long() for comp in coord_components]
        floor_indices = [idx.clamp(0, self.grid_shape[i] - 1) for i, idx in enumerate(floor_indices)]
        ceil_indices = [idx.clamp(0, self.grid_shape[i] - 1) for i, idx in enumerate(ceil_indices)]
        weights = [coord_components[i] - floor_indices[i].float() for i in range(n_dims)]

        # Generate all 2^n corner combinations
        corner_indices = []
        corner_weights = []
        for corner in range(2 ** n_dims):
            idxs = []
            w = torch.ones_like(weights[0])
            for d in range(n_dims):
                if (corner >> d) & 1:
                    idxs.append(ceil_indices[d])
                    w = w * weights[d]
                else:
                    idxs.append(floor_indices[d])
                    w = w * (1 - weights[d])
            corner_indices.append(idxs)
            corner_weights.append(w)

        # Stack indices for advanced indexing
        stacked_indices = [torch.stack([corner_indices[c][d] for c in range(2 ** n_dims)], dim=0) for d in range(n_dims)]
        # shape: (2^n, ...coords.shape)
        stacked_weights = torch.stack(corner_weights, dim=0)  # (2^n, ...coords.shape)

        # Advanced indexing: get features at each corner
        features = self.feature_grid[tuple(stacked_indices)]  # (2^n, ...coords.shape, feature_dim)
        # Weighted sum
        interpolated = torch.sum(features * stacked_weights.unsqueeze(-1), dim=0)
        return interpolated

class RBFInterpolation(Interpolation):
    def __init__(
        self,
        feature_grid: torch.Tensor,
        n_coord_dims: int,
        n_train_stratified_samples_per_dim: int,
        n_test_samples_per_dim: int,
        rbf_type: str,
        dataset: Dataset | None,
        neighborhood_ring_size: int,
        epsilon: float = 1.0,
    ):
        super().__init__(feature_grid)
        self.rbf_type = rbf_type
        self.epsilon = epsilon
        grid_resolution = torch.tensor(self.feature_grid.shape[:n_coord_dims]) # [(grid_resolution, ) * n_coord_dims]
        offsets = torch.stack(torch.meshgrid(*[torch.arange(-neighborhood_ring_size + 1, neighborhood_ring_size + 1) for _ in range(n_coord_dims)]), dim=-1).view(-1, n_coord_dims) # [(2 * neighborhood_ring_size) ** n_coord_dims, n_coord_dims]
        
        def generate_grid_indices(n_samples_per_dim):
            n_repeats = max(0, n_samples_per_dim // (grid_resolution[0].item() - 1))
            grid_indices = [torch.arange(s - 1).repeat_interleave(n_repeats) for s in grid_resolution]
            grid_indices = torch.cartesian_prod(*grid_indices).view(-1, n_coord_dims) # [n_samples_per_dim ** n_coord_dims, n_coord_dims]
            return grid_indices
        
        def generate_neighborhood_indices(grid_indices):
            neighborhood_indices = grid_indices.unsqueeze(-2) + offsets  # [n_samples_per_dim ** n_coord_dims, (2 * neighborhood_ring_size) ** n_coord_dims, n_coord_dims]
            return torch.clamp(neighborhood_indices, 0, grid_resolution[0].item() - 1)[None] # [1, n_samples_per_dim ** n_coord_dims, (2 * neighborhood_ring_size) ** n_coord_dims, n_coord_dims]

        # Compute additional grid indices for dataset coordinates if dataset is provided
        if dataset is not None:
            scaled_coords = (dataset.coords + 1) * 0.5 * (grid_resolution - 1)
            dataset_grid_indices = scaled_coords.long().clamp(0, grid_resolution[0].item() - 1)  # [dataset_size, n_coord_dims]
            self.neighborhood_indices_dataset = generate_neighborhood_indices(dataset_grid_indices)
            
        # Generate neighborhood indices for training and testing
        self.neighborhood_indices_train = generate_neighborhood_indices(generate_grid_indices(n_train_stratified_samples_per_dim))
        self.neighborhood_indices_test = generate_neighborhood_indices(generate_grid_indices(n_test_samples_per_dim))

    def _rbf_weight(self, distances: torch.Tensor) -> torch.Tensor:
        # Remember to choose size of neighborhood_indices based on the RBF type, default (2 * neighborhood_ring_size) ** n_coord_dims holds for only Gaussian RBF
        if self.rbf_type == 'gaussian':
            weights = torch.exp(-(self.epsilon * distances) ** 2)
        elif self.rbf_type == 'inverse_quadratic':
            weights = 1.0 / (1.0 + (self.epsilon * distances) ** 2)
        elif self.rbf_type == 'inverse_multiquadric':
            weights = 1.0 / torch.sqrt(1.0 + (self.epsilon * distances) ** 2)
        else:
            raise ValueError(f'Unknown rbf_type: {self.rbf_type}')
        return weights

    def interpolate(self, coords: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        if kwargs['train']:
            neighborhood_indices = self.neighborhood_indices_train
            if 'random_indices' in kwargs:
                neighborhood_indices = torch.cat((self.neighborhood_indices_dataset[:, kwargs['random_indices'].squeeze()], neighborhood_indices), dim=1)
        else:
            if 'patch_indices' in kwargs and kwargs['patch_indices'] is not None:
                neighborhood_indices = self.neighborhood_indices_test[:, kwargs['patch_indices'].squeeze()]
            else: 
                neighborhood_indices = self.neighborhood_indices_test

        distances = torch.norm(coords.unsqueeze(-2) - neighborhood_indices + 1e-8, dim=-1) # [1, n_samples_per_dim ** n_coord_dims, (2 * neighborhood_ring_size) ** n_coord_dims]
        weights = self._rbf_weight(distances)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        neighborhood_features = self.feature_grid[tuple(neighborhood_indices.unbind(-1))] # [1, n_samples_per_dim ** n_coord_dims, (2 * neighborhood_ring_size) ** n_coord_dims, feature_dim]
        interpolated_features = torch.sum(weights.unsqueeze(-1) * neighborhood_features, dim=-2)

        return interpolated_features # [1, n_samples_per_dim ** n_coord_dims, feature_dim]
