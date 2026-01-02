import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Dict

from .boundary import Boundary
from .interpolation import LinearInterpolation, RBFInterpolation


class SignalRepresentation(nn.Module):
    def __init__(self, boundary: Boundary, in_features: int, out_features: int):
        super().__init__()
        self.boundary = boundary
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, coords: torch.Tensor, **kwargs) -> torch.Tensor:
        normalized_coords = self.boundary.normalization(coords) # [1, n_samples_per_dim ** n_coord_dims, n_coord_dims]
        pred_signal = self._predict(normalized_coords, **kwargs) # [1, n_samples_per_dim ** n_coord_dims, n_signal_dims]
        return self.boundary.dirichlet_condition(pred_signal, coords)

    def _predict(self, coords: torch.Tensor, **kwargs) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError('Subclasses must implement this method.')

class SineLayer(nn.Module):      
    def __init__(self, in_features: int, out_features: int, bias: bool = True, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)        
        self.init_weights()
    
    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))
   
class Siren(SignalRepresentation):
    def __init__(
        self,
        boundary: Boundary,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__(boundary, in_features, out_features)
        self.net = []
        self.net.append(SineLayer(self.in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, self.out_features)            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, self.out_features, is_first=False, omega_0=hidden_omega_0))    
        self.net = nn.Sequential(*self.net)
    
    def _predict(self, coords: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        return self.net(coords)

class PINN(SignalRepresentation):
    def __init__(
        self,
        boundary: Boundary,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int,
    ):
        super().__init__(boundary, in_features, out_features)
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))

        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.GELU())
        
        self.net.append(nn.Linear(hidden_features, hidden_features))
        self.net.append(nn.GELU())

        final_linear = nn.Linear(hidden_features, out_features)
            
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def _predict(self, coords: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        return self.net(coords)

class FeatureGrid(SignalRepresentation):
    def __init__(
        self,
        boundary: Boundary,
        in_features: int,
        out_features: int,
        grid_resolution: int,
        feature_dim: int,
        interpolation_type: str,
        n_train_stratified_samples_per_dim: int,
        n_test_samples_per_dim: int,
        feature_decoder_type: str,
        scales: int,
        dataset: Dataset | None,
        rbf_type: str,
        neighborhood_ring_size: int,
    ):
        super().__init__(boundary, in_features, out_features)
        self.scales = scales
        self.feature_grids = nn.ParameterList([
            nn.Parameter(torch.empty(*(1 + grid_resolution // (2 ** s),) * in_features, feature_dim)) 
            for s in range(scales)
        ])
        self.grid_shapes = torch.stack([torch.tensor(grid.shape[:-1]) for grid in self.feature_grids]).unsqueeze(1).unsqueeze(2)
        for grid in self.feature_grids:
            nn.init.uniform_(grid, a=-1e-4, b=1e-4)

        if feature_decoder_type == 'linear':
            self.linear_feature_decoder(feature_dim * scales)
        elif feature_decoder_type == 'mlp':
            self.mlp_feature_decoder(feature_dim * scales)

        self.interpolations = [
            LinearInterpolation(grid) if interpolation_type == 'lerp'
            else RBFInterpolation(grid, in_features, n_train_stratified_samples_per_dim, n_test_samples_per_dim, rbf_type, dataset, neighborhood_ring_size)
            if interpolation_type == 'rbf'
            else None
            for grid in self.feature_grids
        ]

    def linear_feature_decoder(self, total_feature_dim: int) -> None:
        self.feature_decoder = nn.Linear(total_feature_dim, self.out_features)
        nn.init.xavier_uniform_(self.feature_decoder.weight)
        nn.init.zeros_(self.feature_decoder.bias)

    def mlp_feature_decoder(self, total_feature_dim: int, hidden_dim: int = 64) -> None:
        self.feature_decoder = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.out_features)
        )
        nn.init.xavier_uniform_(self.feature_decoder[0].weight)
        nn.init.zeros_(self.feature_decoder[0].bias)
        nn.init.xavier_uniform_(self.feature_decoder[2].weight)
        nn.init.zeros_(self.feature_decoder[2].bias)
 
    def _predict(self, coords: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        coords_expanded = coords.unsqueeze(0).expand(self.scales, *coords.shape)  # [scales, 1, n_samples_per_dim ** n_coord_dims, n_coord_dims]
        scaled_coords = (coords_expanded + 1) * 0.5 * (self.grid_shapes - 1)  # [scales, 1, n_samples_per_dim ** n_coord_dims, n_coord_dims]
        interpolated_features = torch.cat([interp.interpolate(scaled_coords[i], **kwargs) for i, interp in enumerate(self.interpolations)], dim=-1) # [1, n_samples_per_dim ** n_coord_dims, feature_dim * scales]
        return self.feature_decoder(interpolated_features) # [1, n_samples_per_dim ** n_coord_dims, n_signal_dims]
