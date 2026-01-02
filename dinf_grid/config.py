import torch
import configargparse

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
else:
    device = torch.device("cpu")

def get_config_parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config_filepath', required=True, is_config_file=True, help='Config file path')
    parser.add_argument('-n', '--expt_name', type=str, required=True, help='Experiment name; this will also be the name of subdirectory in logging_dir')
    parser.add_argument('--logging_dir', type=str, default='logs', help='Root directory for logging')

    # Randomness / reproducibility
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed for reproducibility')

    # Experiment parameters
    parser.add_argument('--n_coord_dims', type=int, default=1, help='Number of input coordinate dimensions')
    parser.add_argument('--coord_space', type=float, nargs='+', default=[1], help='Coordinate space limits; e.g. [1] for 1D, [1, 1] for 2D')
    parser.add_argument('--n_signal_dims', type=int, default=1, help='Number of output signal dimensions')
    parser.add_argument('--target_type', type=str, default='linear', choices=['linear', 'cubic', 'sinusoidal', 'image', 'helmholtz', 'neuralclothsim', 'sdf', 'advection', 'heat', 'zalesak'], help="Target to learn; options: 'linear', 'cubic', 'sinusoidal', 'image', 'helmholtz', 'neuralclothsim', 'sdf', 'advection', 'heat', 'zalesak'")
    parser.add_argument('--loss_type', type=str, default='signal', choices=['signal', 'gradient', 'laplacian', 'pde'], help="Type of loss function to use; possible options are: 'signal' when target_type is one of {linear, cubic, sinusoidal, image, sdf}, 'gradient' when target_type is one of {linear, cubic, sinusoidal, image}, 'laplacian' when target_type is one of {linear, cubic, sinusoidal, image}, and 'pde' when target_type is one of {helmholtz, neuralclothsim, advection, heat, zalesak, sdf}")
    parser.add_argument('--boundary_condition', type=str, default='no_boundary', choices=['no_boundary', 'neuralclothsim_origin_fixed', 'neuralclothsim_top_left_fixed', 'neuralclothsim_top_left_top_right_moved', 'advection_boundary', 'heat_boundary', 'zalesak_boundary'], help='Name of the boundary condition; can be one of no_boundary, neuralclothsim_origin_fixed, neuralclothsim_top_left_fixed, neuralclothsim_top_left_top_right_moved advection_boundary, heat_boundary, or zalesak_boundary')

    # Module parameters
    parser.add_argument('--module_type', type=str, default='feature_grid', choices=['siren', 'feature_grid', 'pinn'], help="Type of module to use; options: 'siren', 'feature_grid', 'pinn'; our method is 'feature_grid' whereas 'siren' and 'pinn' are baselines/comparisons")
    
    # Siren parameters
    parser.add_argument('--hidden_features', type=int, default=256, help='Number of hidden features for the Siren module')
    parser.add_argument('--hidden_layers', type=int, default=3, help='Number of hidden layers for the Siren module')
    
    # FeatureGrid parameters
    parser.add_argument('--interpolation_type', type=str, default='rbf', choices=['lerp', 'rbf'], help="Interpolation technique to use for the FeatureGrid module; options: 'lerp', 'rbf'; our method is 'rbf' (radial basis function) whereas 'lerp' (linear interpolation) is a baseline/comparison")
    parser.add_argument('--rbf_type', type=str, default='gaussian', choices=['gaussian', 'inverse_quadratic', 'inverse_multiquadric'], help="RBF type for RBF interpolation: 'gaussian', 'inverse_quadratic', 'inverse_multiquadric'")
    parser.add_argument('--neighborhood_ring_size', type=int, default=3, help='Neighborhood ring size for RBF interpolation')
    parser.add_argument('--grid_resolution', type=int, default=128, help='Resolution of the feature grid for the FeatureGrid module')
    parser.add_argument('--scales', type=int, default=1, help='Number of scales for multi-scale interpolation in FeatureGrid')
    parser.add_argument('--feature_dim', type=int, default=3, help='Dimensionality of the feature grid for the FeatureGrid module; in practice, setting feature_dim=n_signal_dims works well for most tasks')
    parser.add_argument('--feature_decoder_type', type=str, default='linear', choices=['linear', 'mlp'], help="Type of feature decoder to use in FeatureGrid; options: 'linear', 'mlp'; 'linear' is a single linear layer, whereas 'mlp' is a 2-layer MLP with Tanh activation, linear works very well for most tasks")

    # Training options
    parser.add_argument('--sampling_strategy', type=str, default='stratified', choices=['data', 'stratified', 'data_and_stratified'], help='Sampling strategy for training data; if data is available (e.g. for image, sdf targets), go with data or data_and_stratified, else stratified (for all other \'target_type\'s); data sampling uses only data points, stratified sampling uses only uniformly sampled points in the coordinate space, and data_and_stratified uses both data points and uniformly sampled points')
    parser.add_argument('--n_train_stratified_samples_per_dim', type=int, default=0, help='Number of training samples per dimension for stratified sampling')
    parser.add_argument('--n_train_data_samples_per_dim', type=int, help='Number of data samples to draw per iteration when using data sampling or data_and_stratified sampling')
    
    parser.add_argument('--lrate', type=float, default=5e-6, help='Learning rate; set this appropriately based on the loss_type and target_type')
    parser.add_argument('--decay_lrate', action='store_true', default=True, help='Whether to decay learning rate')
    parser.add_argument('--lrate_decay_steps', type=int, default=5000, help='Learning rate decay steps')
    parser.add_argument('--lrate_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    
    parser.add_argument('--n_iterations', type=int, default=5001, help='Total number of training iterations')
    parser.add_argument('--i_weights', type=int, default=400, help='Frequency of saving weights as checkpoints')

    # Reload options
    parser.add_argument('--no_reload', action='store_true', default=False, help='Do not resume training from checkpoint, rather train from scratch; Default behavior is to resume training: either from i_ckpt if specified or from last saved checkpoint')
    parser.add_argument('--i_ckpt', type=int, help='Weight checkpoint to reload for resuming training or for performing evaluation (see \'test_only\' option below ); if None, the latest checkpoint is used')    

    # Testing options
    parser.add_argument('--i_test', type=int, default=100, help='Frequency of evaluating the model and logging results to TensorBoard during training')
    parser.add_argument('--test_only', action='store_true', help='Evaluate model from i_ckpt (or from last saved checkpoint) and log results to TensorBoard; do not resume training')
    parser.add_argument('--n_test_samples_per_dim', type=int, default=20, help='Number of testing samples per coordinate dimension')

    # Target specific options
    # Image target options
    parser.add_argument('--image_path', type=str, default=None, help='Path to input image for image target')
    parser.add_argument('--use_patch', action='store_true', default=False, help='Whether to use patch-based visualisation for image target; this is useful for high-resolution images')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size to use for patch-based visualisation for image target; used only if use_patch is True')

    # Helmholtz target options
    parser.add_argument('--wavenumber', type=float, default=20.0, help='Wavenumber for Helmholtz equation experiments')
    
    # SDF target options
    parser.add_argument('--point_cloud_path', type=str, default='assets/dragon_vrip.ply', help='Path to input .ply file for SDF target; if loss_type is signal, make sure the ply file is a mesh with faces, otherwise if loss_type pde ensure that it has normals')

    # Advection target options
    parser.add_argument('--advection_mu', type=float, nargs='+', default=[-1.5], help='Mu (mean) for advection initial Gaussian wave')
    parser.add_argument('--advection_sigma', type=float, nargs='+', default=[0.1], help='Sigma (standard deviation) for advection initial Gaussian wave')
    parser.add_argument('--advection_velocity', type=float, nargs='+', default=[0.25], help='Velocity for advection equation')

    # Zalesak target options
    parser.add_argument('--zalesak_radius', type=float, default=0.5, help="Radius of the slotted disk in Zalesak's problem")
    parser.add_argument('--zalesak_slot_width', type=float, default=0.1, help="Slot width of the slotted disk in Zalesak's problem")
    parser.add_argument('--zalesak_slot_height', type=float, default=0.25, help="Slot height of the slotted disk in Zalesak's problem")
    parser.add_argument('--zalesak_center_x', type=float, default=0.0, help="X center of the slotted disk in Zalesak's problem")
    parser.add_argument('--zalesak_center_y', type=float, default=0.0, help="Y center of the slotted disk in Zalesak's problem")

    return parser
