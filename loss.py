import torch
from torch.nn.functional import l1_loss, mse_loss
from utils.diff_operators import jacobian, laplacian
from target import Target

class Loss:
    def __init__(self, loss_type: str, target: Target):
        self.loss_type = loss_type
        self.target = target
        self.loss_fn = mse_loss if self.target.target_type in ['image'] else l1_loss

    def __call__(self, pred_signal: torch.Tensor, coords: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.loss_type == 'pde':
            loss = self.target.compute(coords, pred_signal, **kwargs)
        else:
            gt_signal, gt_gradient, gt_laplacian = self.target.compute(coords, **kwargs)
            match self.loss_type:
                case 'signal':
                    loss = self.loss_fn(pred_signal, gt_signal)
                case 'gradient':
                    pred_gradient = jacobian(pred_signal, coords)  # [1, n_train_samples_per_dim ** n_coord_dims, n_coord_dims]
                    loss = self.loss_fn(pred_gradient, gt_gradient)
                case 'laplacian':
                    pred_gradient = jacobian(pred_signal, coords)
                    pred_laplacian = laplacian(pred_gradient, coords)  # [1, n_train_samples_per_dim ** n_coord_dims, 1]
                    loss = self.loss_fn(pred_laplacian, gt_laplacian)
        return loss