import torch
from torch.autograd import grad

def jacobian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ''' jacobian of y wrt x '''
    batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(batch_size, num_observations, y.shape[-1], x.shape[-1])
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        gradient = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True, allow_unused=True)[0]
        if gradient is not None:
            jac[:, :, i, :] = gradient
    return jac

def laplacian(gradient: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    '''
    Computes the laplacian of the given gradient with respect to the coordinates.
    Assumes the gradient is a tensor of shape (..., n_coords).
    '''
    laplacian = torch.zeros_like(gradient[..., 0])  # Initialize laplacian
    for signal_dim in range(gradient.shape[-2]):  # Iterate over each signal dimension
        for coord_dim in range(coords.shape[-1]):  # Iterate over each coordinate dimension
            grad_dim = jacobian(gradient[..., signal_dim, coord_dim].unsqueeze(-1), coords)[..., coord_dim]  # Second derivative w.r.t. each dimension
            laplacian[..., signal_dim] += grad_dim[...,0]
    return laplacian