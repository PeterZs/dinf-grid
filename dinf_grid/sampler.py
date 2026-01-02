import torch
from typing import Dict, Tuple, Union
from torch.utils.data import Dataset


def get_mgrid(sidelen: Union[Tuple[int, ...], int], stratified: bool = False, dim: int = 2) -> torch.Tensor:
    # Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    if isinstance(sidelen, int):
        sidelen = (sidelen,)
    if len(sidelen) != dim:
        raise ValueError(f'sidelen must have {dim} entries, got {len(sidelen)}')

    sidelen = tuple(sidelen)
    denominators = [s if stratified else s - 1 for s in sidelen]

    axes = []
    for size, denom in zip(sidelen, denominators):
        axis = torch.arange(size)
        axis = axis / axis.new_tensor(float(denom))
        axes.append(axis)

    mesh = torch.meshgrid(*axes, indexing='ij')
    grid_coords = torch.stack(mesh, dim=-1)
    grid_coords -= 0.5
    grid_coords *= 2.0
    return grid_coords.view(-1, dim)
    
class StratifiedGridSampler(Dataset):
    def __init__(self, n_stratified_samples_per_dim: int, n_coord_dims: int, coord_space: Tuple[float, ...]):
        self.n_stratified_samples_per_dim = n_stratified_samples_per_dim
        self.n_coord_dims = n_coord_dims
        self.coord_space = coord_space
        self.n_stratified_samples = n_stratified_samples_per_dim ** n_coord_dims
        # Assume coords[..., 0]=t, coords[..., 1]=x (or y,z...)
        self.cell_coords = get_mgrid(n_coord_dims * (n_stratified_samples_per_dim,), stratified=True, dim=n_coord_dims)

    def __len__(self):
        return self.n_stratified_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        coords = self.cell_coords.clone()
        coords += torch.rand([self.n_stratified_samples, self.n_coord_dims]) * 2 / self.n_stratified_samples_per_dim
        for i in range(self.n_coord_dims):
            coords[..., i] *= self.coord_space[i]
        return coords.requires_grad_(True), {}

class DataSampler(Dataset):
    def __init__(self, n_data_samples_per_dim: int, n_coord_dims: int, dataset: Dataset):        
        self.n_data_samples = n_data_samples_per_dim ** n_coord_dims
        self.dataset = dataset

    def __len__(self):
        return self.n_data_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        random_indices = torch.randint(0, len(self.dataset), (self.n_data_samples,))
        coords = self.dataset.get_coords(random_indices)
        return coords.requires_grad_(True), {'random_indices': random_indices}

class DataandStratifiedSampler(Dataset):
    def __init__(self, stratified_sampler: StratifiedGridSampler, data_sampler: DataSampler):
        self.stratified_sampler = stratified_sampler
        self.data_sampler = data_sampler

    def __len__(self):
        return len(self.stratified_sampler) + len(self.data_sampler)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        on_dataset_coords, sample_args = self.data_sampler[0]
        on_stratified_grid_coords, _ = self.stratified_sampler[0]
        coords = torch.cat((on_dataset_coords, on_stratified_grid_coords), dim=0)
        return coords, sample_args
