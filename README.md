# <em>∂</em><sup>∞</sup>-Grid: A Neural Differential Equation Solver with Differentiable Feature Grids

[![arXiv](https://img.shields.io/badge/arXiv-2601.10715-b31b1b)](https://arxiv.org/abs/2601.10715)
[![Project Page](https://img.shields.io/badge/Project-Page-4D8DC9)](https://4dqv.mpi-inf.mpg.de/DInf-Grid/)
<!--[![Video](https://img.shields.io/badge/Video-YouTube-FF0000)](#)-->

[Navami Kairanda](https://people.mpi-inf.mpg.de/~nkairand/),
[Shanthika Naik](https://shanthika.github.io/),
[Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/),
[Avinash Sharma](https://3dcomputervision.github.io/index.html),
[Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/),
[Vladislav Golyanik](https://people.mpi-inf.mpg.de/~golyanik/)  
Max Planck Institute for Informatics <!-- and collaborators in ICLR 2026-->

This repository contains the official implementation of the paper "<em>∂</em><sup>∞</sup>-Grid: Differentiable Grid Representations for Fast and Accurate Solutions to Differential Equations".

<!-- Video/Teaser -->

## What is <em>∂</em><sup>∞</sup>-Grid?
*We present a novel differentiable grid-based representation for efficiently solving differential equations (DEs). Widely used architectures for neural solvers, such as sinusoidal neural networks, are coordinate-based MLPs that are, both, computationally intensive and slow to train. Although grid-based alternatives for implicit representations (e.g., Instant-NGP and K-Planes) train faster by exploiting signal structure, their reliance on linear interpolation restricts their ability to compute higher-order derivatives, rendering them unsuitable for solving DEs. In contrast, our approach overcomes these limitations by combining the efficiency of feature grids with radial basis function interpolation, which is infinitely often differentiable. To effectively capture high-frequency solutions and enable stable and faster computation of global gradients, we introduce a multi-resolution decomposition with co-located grids. Our proposed representation, <em>∂</em><sup>∞</sup>-Grid, is trained implicitly using the differential equations as loss functions, enabling accurate modeling of physical fields. We validate <em>∂</em><sup>∞</sup>-Grid on a variety of tasks, including Poisson equation for image reconstruction, the Helmholtz equation for wave fields, and the Kirchhoff-Love boundary value problem for cloth simulation. Our results demonstrate a 5–20x speed-up over coordinate-based MLP-based methods, solving differential equations in seconds or minutes while maintaining comparable accuracy and compactness.*

## News
* [18 Jan 2026] We have released the source code.
* [26 Jan 2026] Our paper has been accepted at ICLR 2026!-->

## Installation

Clone this repository, then choose how you want to set up the environment:

[![pip](https://img.shields.io/badge/install-pip/uv-3775A9)](#pip-installation)
[![conda](https://img.shields.io/badge/install-conda-3E4E7E)](#conda-installation)

<a id="pip-installation"></a>

We recommend using `uv` (fast Python package manager) with a virtual environment:

```bash
uv venv dinf-grid
source dinf-grid/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r requirements.txt
```
<!--
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install tqdm configargparse scikit-image matplotlib trimesh cmapy plyfile tensorboard pysdf
-->

We use the PyTorch CUDA index for `torch` and the default PyPI index for the remaining packages.

<!--

```bash
conda create -n dinf-grid python
conda activate dinf-grid

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install tqdm configargparse scikit-image matplotlib trimesh plyfile tensorboard pysdf
pip install cmapy markupsafe==2.1.5
```
-->
<details>
<summary><a id="conda-installation"></a>If you prefer Conda, click to show instructions</summary>

```bash
conda create -n dinf-grid python
conda activate dinf-grid

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda env create -f environment.yml
```

</details>

## Quickstart
<!--
### High-level overview

The main components of the codebase are:
The codebase has the following structure:

- [config/](config): experiment configuration files (`.ini`) specifying target type (signal, PDE), model choice (Feature grid, Siren, PINN), sampling strategy, and logging settings.
- [run.py](run.py): entry point that parses a config file, builds the model and target, runs training, evaluation, and checkpointing.
- [modules.py](modules.py): implementations of the different interpolation modules (SIREN MLP, grid-based feature interpolation, PINN).
- [interpolation.py](modules.py): implementations of the different interpolation modules (SIREN MLP, grid-based feature interpolation, PINN).
- [sampler.py](sampler.py): stratified grid samplers and data samplers used during training.
- [boundary.py](boundary.py): boundary condition handling for coordinates (e.g., fixed corners, PDE boundaries).
- [dinf_grid/target.py](dinf_grid/target.py): wraps analytical functions, images, and PDE problems into a common interface.
- [loss.py](loss.py): loss definitions for signal, gradient, Laplacian, and PDE objectives.
- [test.py](test.py): evaluation and visualisation routines used during training and at checkpoints.
- [neuralclothsim/](neuralclothsim): X
-->

1. From the project root, run an experiment with:
   
   ```bash
   python run.py -c <config_path> -n <experiment_name>
   ```

   The core training loop is implemented in [run.py](run.py) and configured via `.ini` files in [config/](config).

   For example, to solve the Helmholtz equation:

   ```bash
   python run.py -c config/grid_rbf_pde_helmholtz.ini -n grid_rbf_pde_helmholtz
   ```

   To fit a colour image signal:

   ```bash
   python run.py -c config/image_colour/grid_rbf_signal_image.ini -n grid_rbf_signal_image
   ```

   See the full list of supported experiments in the tables for [PDE objectives](#experiments-with-pde-objectives) and [signal-based objectives](#experiments-with-signal-based-objectives) below. Our training code was tested on single-GPU runs with NVIDIA H100/A100, but you can typically get away with lower GPU memory as well; see the FAQ below for tips on reducing memory usage.

2. Experiments write logs, checkpoints, and TensorBoard visualisations into the directory specified by `logging_dir` in the config file (by default [logs/](logs)). To monitor training metrics and inspect PDE solution visualisations with TensorBoard:

   ```bash
   tensorboard --logdir <logging_dir>
   ```

3. To resume training from a checkpoint:

   ```bash
   python run.py -c <config_path> -n <experiment_name> --i_ckpt <trained_checkpoint>
   ```

4. To run evaluation on a trained checkpoint (logged to Tensorboard):

   ```bash
   python run.py -c <config_path> -n <experiment_name> --test_only --i_ckpt <trained_checkpoint>
   ```

   Omit `--i_ckpt <trained_checkpoint>` if you want to resume or evaluate from the last saved checkpoint.

### Command-line arguments

<details>
<summary>Show key arguments</summary>

- **`-c, --config_filepath`** – Config file path.
- **`-n, --expt_name`** – Experiment name; this will also be the name of subdirectory in logging_dir.
- **`--logging_dir`** – Root directory for logging.
- **`--n_coord_dims`** – Number of input coordinate dimensions.
- **`--coord_space`** – Coordinate space limits; e.g. [1] for 1D, [1, 1] for 2D.
- **`--n_signal_dims`** – Number of output signal dimensions.
- **`--target_type`** – Target to learn; options: 'linear', 'cubic', 'sinusoidal', 'image', 'helmholtz', 'neuralclothsim', 'sdf', 'advection', 'heat', 'zalesak'.
- **`--loss_type`** – Type of loss function to use; possible options are: 'signal' when target_type is one of {linear, cubic, sinusoidal, image, sdf}, 'gradient' when target_type is one of {linear, cubic, sinusoidal, image}, 'laplacian' when target_type is one of {linear, cubic, sinusoidal, image}, and 'pde' when target_type is one of {helmholtz, neuralclothsim, advection, heat, zalesak, sdf}.
 - **`--boundary_condition`** – Name of the boundary condition; can be one of no_boundary, neuralclothsim_origin_fixed, neuralclothsim_top_left_fixed, neuralclothsim_top_left_top_right_moved advection_boundary, heat_boundary, or zalesak_boundary.
- **`--module_type`** – Type of module to use; options: 'siren', 'feature_grid', 'pinn'; our method is 'feature_grid', whereas 'siren' and 'pinn' are baselines/comparisons.
- **`--interpolation_type`** – Interpolation technique to use for the FeatureGrid module; options: 'lerp', 'rbf'; our method is 'rbf' (radial basis function), whereas 'lerp' (linear interpolation) is a baseline/comparison.
- **`--rbf_type`** – RBF type for RBF interpolation: 'gaussian', 'inverse_quadratic', 'inverse_multiquadric'.
- **`--neighborhood_ring_size`** – Neighborhood ring size for RBF interpolation.
- **`--grid_resolution`** – Resolution of the feature grid for the FeatureGrid module.
- **`--scales`** – Number of scales for multi-scale interpolation in FeatureGrid.
- **`--feature_dim`** – Dimensionality of the feature grid for the FeatureGrid module; in practice, setting feature_dim=n_signal_dims works well for most tasks.
- **`--feature_decoder_type`** – Type of feature decoder to use in FeatureGrid; options: 'linear', 'mlp'; 'linear' is a single linear layer, whereas 'mlp' is a 2-layer MLP with Tanh activation, linear works very well for most tasks.
- **`--sampling_strategy`** – Sampling strategy for training data; if data is available (e.g. for image, sdf targets), go with data or data_and_stratified, else stratified (for all other 'target_type's); data sampling uses only data points, stratified sampling uses only uniformly sampled points in the coordinate space, and data_and_stratified uses both data points and uniformly sampled points.
- **`--n_train_stratified_samples_per_dim`** – Number of training samples per dimension for stratified sampling.
- **`--n_train_data_samples_per_dim`** – Number of data samples to draw per iteration when using data sampling or data_and_stratified sampling.
- **`--n_test_samples_per_dim`** – Number of testing samples per coordinate dimension.
- **`--image_path`** – Path to input image for image target.
- **`--point_cloud_path`** – Path to input .ply file for SDF target; if loss_type is signal, make sure the ply file is a mesh with faces, otherwise if loss_type pde ensure that it has normals.

</details>

The full list of command-line arguments and their defaults can be obtained with `python run.py --help`.

### Experiments with PDE objectives

The following table summarises common PDE-style experiments (including gradient and Laplacian objectives), i.e., runs with `loss_type` in `{gradient, laplacian, pde}`.

| Task / Equation        | `target_type` | Config file                                                      |
|------------------------------------|---------------|------------------------------------------------------------------|
| Poisson - gradient (grayscale image)        | `image`       | [image_grayscale/grid_rbf_gradient_image.ini](config/image_grayscale/grid_rbf_gradient_image.ini) |
| Poisson - gradient (colour image)              | `image`       | [image_colour/grid_rbf_gradient_image.ini](config/image_colour/grid_rbf_gradient_image.ini) |
| Poisson - Laplacian (grayscale image)       | `image`       | [image_grayscale/grid_rbf_laplacian_image.ini](config/image_grayscale/grid_rbf_laplacian_image.ini) |
| Poisson - Laplacian (colour image)             | `image`       | [image_colour/grid_rbf_laplacian_image.ini](config/image_colour/grid_rbf_laplacian_image.ini) |
| Eikonal (SDF from oriented point cloud) | `sdf`         | [grid_rbf_pde_sdf.ini](config/grid_rbf_pde_sdf.ini)             |
| Heat (diffusion of sine wave)          | `heat`        | [grid_rbf_pde_heat.ini](config/grid_rbf_pde_heat.ini)           |
| Advection (1D Gaussian wave)       | `advection`   | [grid_rbf_pde_advection.ini](config/grid_rbf_pde_advection.ini) |
| Advection (2D Gaussian wave)       | `advection`   | [grid_rbf_pde_advection_2d.ini](config/grid_rbf_pde_advection_2d.ini) |
| Advection (Zalesak’s disk)         | `zalesak`     | [grid_rbf_pde_zalesak.ini](config/grid_rbf_pde_zalesak.ini)     |
| Helmholtz (2D wave field)          | `helmholtz`   | [grid_rbf_pde_helmholtz.ini](config/grid_rbf_pde_helmholtz.ini) |
| Kirchhoff-Love (cloth simulation)  | `neuralclothsim` | [grid_rbf_pde_neuralclothsim.ini](config/grid_rbf_pde_neuralclothsim.ini) |

Image and point-cloud paths can be controlled with the `--image_path <image_path>` and `--point_cloud_path <point_cloud_path>` arguments. For other experiment-specific configurations, see [dinf_grid/config.py](dinf_grid/config.py) and [dinf_grid/target.py](dinf_grid/target.py).

### Experiments with signal-based objectives

These runs overfit to a given signal (image or surface), using `loss_type = 'signal'`.

| Task            | `target_type` | Config file                                                      |
|----------------------------|---------------|------------------------------------------------------------------|
| Image (grayscale)          | `image`       | [image_grayscale/grid_rbf_signal_image.ini](config/image_grayscale/grid_rbf_signal_image.ini) |
| Image (colour)             | `image`       | [image_colour/grid_rbf_signal_image.ini](config/image_colour/grid_rbf_signal_image.ini) |
| SDF from mesh surface      | `sdf`         | [grid_rbf_signal_sdf.ini](config/grid_rbf_signal_sdf.ini)       |

## How to solve your own PDEs with <em>∂</em><sup>∞</sup>-Grid?

The easiest way to add a new problem is to start from an existing experiment that is closest to what you want (see the tables above), copy its config file, and then adapt the steps below.

1. Create or edit a config in `config/grid_rbf_pde_{target_type}`.
   - Set the field/signal definition with `n_coord_dims`, `coord_space`, `n_signal_dims`.
   - Choose an appropriate `sampling_strategy` (data, stratified, or data_and_stratified).
2. Implement or adjust the target in `target.py` for your PDE or signal.
3. Add any hard boundary conditions in `boundary.py` (optional).
4. Extend testing and metrics in `test.py` for problem specific evaluation.
5. Add TensorBoard visualisations and metrics in `logger.py`.

Once these pieces are wired up, you can launch your new problem with `python run.py -c <your_config.ini> -n <experiment_name>` and monitor training via TensorBoard.

## FAQ

- *Training runs out of memory or is too slow.* Reduce `n_train_stratified_samples_per_dim`, `n_train_data_samples_per_dim`, or `n_test_samples_per_dim` in the config file, and/or lower the feature grid size `grid_resolution`. You can also set `feature_dim` to `n_signal_dims`, `neighborhood_ring_size` to 2, and `scales` to 1. See the paper appendices for how these settings affect results.
- *How do I run comparisons/baselines?* Our method uses `module_type=feature_grid` with `interpolation_type=rbf`. For baselines (Siren, PINN, or a linear-interpolated grid), set `module_type` to `siren`, `pinn`, or `feature_grid`, and set `interpolation_type` to `lerp` for the linear grid.
   ```bash
   python run.py -c config/grid_rbf_pde_neuralclothsim.ini -n grid_lerp_pde_neuralclothsim --interpolation_type lerp # Feature grid with linear interpolation
   python run.py -c config/siren_pde_neuralclothsim.ini # Siren
   python run.py -c config/siren_pde_helmholtz.ini -n pinn_pde_helmholtz --module_type pinn # PINN with GELU activation
   ```
<!--python run.py -c config/mlp_pde_neuralclothsim.ini -n siren_pde_neuralclothsim --module_type siren
python run.py -c config/mlp_pde_helmholtz.ini -n pinn_pde_helmholtz --module_type pinn-->

## Acknowledgements

This project builds upon the following excellent open source repositories:

* [Siren](https://github.com/vsitzmann/siren) - An implicit neural representation that leverages periodic (sinusoidal) activation functions.
* [NeuralClothSim](https://github.com/navamikairanda/neuralclothsim) - A quasistatic cloth simulator using thin shells, in which surface deformation is encoded in neural network weights in the form of a neural field.

We thank the authors of these projects.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{kairanda2026partialinfgrid,
   title   = {${\partial^\infty}$-Grid: A Neural Differential Equation Solver with Differentiable Feature Grids},
   author  = {Kairanda, Navami and Naik, Shanthika and Habermann, Marc and Sharma, Avinash and Theobalt, Christian and Golyanik, Vladislav},
   year    = {2026},
   journal = {International Conference on Learning Representations}
}
```

## License

This software is provided freely for non-commercial use. We release this code under the MIT license, which you can find in the file LICENSE.
