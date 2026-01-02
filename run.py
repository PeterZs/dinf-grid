import os
import torch
from tqdm import trange
from shutil import copyfile
from torch.utils.data import DataLoader

from utils.ops import count_parameters
from dinf_grid.logger import get_logger, TensorBoardLogger
from dinf_grid.config import get_config_parser, device

from dinf_grid.sampler import StratifiedGridSampler, DataSampler, DataandStratifiedSampler
from dinf_grid.boundary import Boundary
from dinf_grid.modules import Siren, FeatureGrid, PINN
from dinf_grid.loss import Loss
from dinf_grid.target import Target
from dinf_grid.test import Test

def train():
    args = get_config_parser().parse_args()
    torch.manual_seed(args.random_seed)
    log_dir = os.path.join(args.logging_dir, args.expt_name)
    weights_dir = os.path.join(log_dir, 'weights')
        
    for dir in [log_dir, weights_dir]:
        os.makedirs(dir, exist_ok=True)
            
    logger = get_logger(log_dir, args.expt_name)
    logger.info(args)

    tb_logger = TensorBoardLogger(log_dir, loss_type=args.loss_type, target_type=args.target_type)
    tb_logger.writer.add_text('args', str(args))

    copyfile(args.config_filepath, os.path.join(log_dir, f'{args.expt_name}.ini'))

    args.coord_space = [1] * args.n_coord_dims if args.coord_space == [1] else args.coord_space
    
    target = Target(args)    
    boundary = Boundary(args.boundary_condition, args.coord_space, target)
    
    if args.module_type == 'feature_grid':
        if args.n_test_samples_per_dim % args.grid_resolution != 0 or args.n_train_stratified_samples_per_dim % args.grid_resolution != 0:
            raise ValueError("n_test_samples_per_dim and n_train_stratified_samples_per_dim must be a multiple of grid_resolution when using feature_grid.")
        module = FeatureGrid(
            boundary,
            in_features=args.n_coord_dims,
            out_features=args.n_signal_dims,
            grid_resolution=args.grid_resolution,
            feature_dim=args.feature_dim,
            interpolation_type=args.interpolation_type,
            n_train_stratified_samples_per_dim=args.n_train_stratified_samples_per_dim,
            n_test_samples_per_dim=args.n_test_samples_per_dim,
            feature_decoder_type=args.feature_decoder_type,
            scales=args.scales, 
            dataset=target.dataset,
            rbf_type=args.rbf_type,
            neighborhood_ring_size=args.neighborhood_ring_size,
        ).to(device)
    elif args.module_type == 'siren':
        module = Siren(
            boundary,
            in_features=args.n_coord_dims,
            out_features=args.n_signal_dims,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers
        ).to(device)
    elif args.module_type == 'pinn':
        module = PINN(
            boundary,
            in_features=args.n_coord_dims,
            out_features=args.n_signal_dims,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers
        ).to(device)

    num_params = count_parameters(module)
    logger.info(f'{args.module_type.capitalize()} module initialized with {num_params} trainable parameters.')
    tb_logger.writer.add_text('num_parameters', f'{num_params} trainable parameters')

    optimizer = torch.optim.Adam(lr=args.lrate, params=module.parameters())
   
    test = Test(module, target, args.n_test_samples_per_dim, args.n_coord_dims, args.coord_space, tb_logger)
    
    if args.i_ckpt is not None:
        ckpts = [os.path.join(weights_dir, f'{args.i_ckpt:06d}.tar')]
    else:
        ckpts = [os.path.join(weights_dir, f) for f in sorted(os.listdir(weights_dir)) if '0.tar' in f]            
    logger.info(f'Found ckpts: {ckpts}')
    if len(ckpts) > 0 and (args.test_only or not args.no_reload):        
        ckpt = torch.load(ckpts[-1])
        module.load_state_dict(ckpt['module_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if args.test_only:
            logger.info(f'Evaluating {args.expt_name} from checkpoint {ckpts[-1]}')
            test(ckpt['global_step'])
            return
        
    if len(ckpts)==0 or args.no_reload:
        logger.info(f'Starting experiment {args.expt_name}')
        global_step = 0
    else:
        logger.info(f'Resuming experiment {args.expt_name} from checkpoint {ckpts[-1]}')
        global_step = ckpt['global_step']
    
    loss_fn = Loss(loss_type=args.loss_type, target=target)

    match args.sampling_strategy:
        case 'data':
            sampler = DataSampler(args.n_train_data_samples_per_dim, args.n_coord_dims, target.dataset)
        case 'stratified':
            sampler = StratifiedGridSampler(args.n_train_stratified_samples_per_dim, args.n_coord_dims, coord_space=args.coord_space)
        case 'data_and_stratified':
            data_sampler = DataSampler(args.n_train_data_samples_per_dim, args.n_coord_dims, target.dataset)
            stratified_sampler = StratifiedGridSampler(args.n_train_stratified_samples_per_dim, args.n_coord_dims, coord_space=args.coord_space)
            sampler = DataandStratifiedSampler(stratified_sampler, data_sampler)
    
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
    
    iter_start_event = torch.cuda.Event(enable_timing=True)
    iter_end_event = torch.cuda.Event(enable_timing=True)

    for i in trange(global_step, args.n_iterations):
        iter_start_event.record()
        
        coords, kwargs = next(iter(dataloader))
        pred_signal = module(coords, train=True, **kwargs)
        loss = loss_fn(pred_signal, coords, **kwargs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iter_end_event.record()
                
        torch.cuda.synchronize()
        iteration_time_in_seconds = iter_start_event.elapsed_time(iter_end_event) / 1000.0
        tb_logger.writer.add_scalar('time/secs per iteration', iteration_time_in_seconds, i)
        tb_logger.writer.add_scalar('time/iterations per sec', 1 / iteration_time_in_seconds, i)
        tb_logger.writer.add_scalar(f'loss/{args.loss_type}_loss', loss, i)

        if args.decay_lrate:
            new_lrate = args.lrate * args.lrate_decay_rate ** (i / args.lrate_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

        if not i % args.i_weights and i > 0:
            torch.save({
                'global_step': i,
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(weights_dir, f'{i:06d}.tar'))

        if not i % args.i_test:
            logger.info(f'Iteration: {i}, loss: {loss}')
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            tb_logger.writer.add_scalar('cuda/memory_reserved_GB', memory_reserved, i)
            test(i)

    tb_logger.close()

if __name__ == '__main__':
    train()