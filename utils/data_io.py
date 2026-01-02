import numpy as np
import torch
import math
from matplotlib import colors
import matplotlib.pyplot as plt
import cv2
import cmapy
import trimesh
import plyfile
import skimage.measure
from typing import NamedTuple

class Mesh(NamedTuple):
    verts: torch.Tensor
    faces: torch.Tensor
    curvilinear_coords: torch.Tensor
    
def to_uint8(x):
    return (255. * x).astype(np.uint8)

def scale_percentile(pred, min_perc=1, max_perc=99):
    min = np.percentile(pred.cpu().numpy(), min_perc)
    max = np.percentile(pred.cpu().numpy(), max_perc)
    pred = torch.clamp(pred, min, max)
    return (pred - min) / (max - min)

def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if mode == 'scale':
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif mode == 'clamp':
        x = torch.clamp(x, 0, 1)
    return x

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

def grads2img(gradients): # [1, num_samples, channels=2]
    gradients = lin2img(gradients)
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row, cols, 2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1) # [channels=3, height, width]

def laplace2img(laplacian, perc=2): # [batch_size, num_samples, channels=1]
    rescaled_img = rescale_img(lin2img(laplacian), perc=perc).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    uint8_img = to_uint8(rescaled_img)
    colored_img = cv2.applyColorMap(uint8_img, cmapy.cmap('RdBu'))
    rgb_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb_img).permute(2, 0, 1).float() # [channels=3, height, width]

def helmholtz2img(signal):
    uint8_img = to_uint8(signal.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy())
    colored_img = cv2.applyColorMap(uint8_img, cmapy.cmap('RdBu'))
    rgb_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb_img).permute(2, 0, 1).float()

def get_plot_1d_signal(y_pred, y_gt=None, label_pred='Prediction', label_gt='Ground Truth', x=None):
    fig = plt.figure()
    if x is None:
        x = torch.linspace(-1, 1, steps=y_pred.shape[1])
    plt.scatter(x.detach().cpu(), y_pred.detach().cpu(), alpha=0.5, label=label_pred)
    if y_gt is not None:
        plt.scatter(x.detach().cpu(), y_gt.detach().cpu(), alpha=0.5, label=label_gt)
    plt.xlabel('Coords')
    plt.ylabel('Values')
    plt.title('Coords vs Values')
    plt.legend()
    return fig

# Both functions below assume the tensor input is [row*cols, 1]. They expect the samples to be nicely arranged in a grid.
def get_plot_single_tensor(tensor):
    fig = plt.figure()
    ax = fig.gca()
    spatial_sidelen = math.isqrt(tensor.squeeze().shape[0])
    pcolormesh = ax.pcolormesh(tensor.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh, ax=ax)
    plt.close(fig)
    return fig

def get_plot_grid_tensor(tensor_1, tensor_2, tensor_3, tensor_4):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0.22)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    spatial_sidelen = math.isqrt(tensor_1.squeeze().shape[0])
    pcolormesh1 = ax1.pcolormesh(tensor_1.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh2 = ax2.pcolormesh(tensor_2.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh3 = ax3.pcolormesh(tensor_3.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh4 = ax4.pcolormesh(tensor_4.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh1, ax=ax1)
    fig.colorbar(pcolormesh2, ax=ax2)
    fig.colorbar(pcolormesh3, ax=ax3)
    fig.colorbar(pcolormesh4, ax=ax4)
    plt.close(fig)
    return fig

def make_contour_plot(array_2d, mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig

def generate_mesh_topology(spatial_sidelen):
    rows = cols = spatial_sidelen
    last_face_index = cols * (rows - 1)
    
    first_face_bl = [0, cols, 1]  
    first_face_tr = [cols + 1, 1, cols]  
    all_faces = []
    for first_face in [first_face_bl, first_face_tr]:
        last_face = [i + last_face_index - 1 for i in first_face]
        faces = np.linspace(first_face, last_face, last_face_index)
        faces = np.reshape(faces, (rows - 1, cols, 3))
        faces = np.delete(faces, cols - 1, 1)
        faces = np.reshape(faces, (-1, 3))   
        all_faces.append(faces)
    return np.concatenate(all_faces, axis=0)

def save_mesh(filepath, vertices, faces):
    vertices_np = vertices.squeeze().cpu().numpy()
    faces_np = faces.squeeze().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    mesh.export(filepath)

'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF'''
def create_mesh(module, filename, N, offset=None, scale=None):    
    ply_filename = filename

    module.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    num_samples = N ** 3
    overall_index = torch.arange(0, num_samples, 1) # out=torch.LongTensor()
    samples = torch.zeros(num_samples, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False
    '''
    max_batch = 64 ** 3
    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (decoder(sample_subset, train=False).squeeze().detach().cpu())
        head += max_batch

    sdf_values = samples[:, 3]
    '''
    sdf_values = (module(samples[None, ...,:3], train=False).squeeze().detach().cpu())
    sdf_values = sdf_values.reshape(N, N, N)

    convert_sdf_samples_to_ply(sdf_values.data.cpu(), voxel_origin, voxel_size, ply_filename + ".ply", offset, scale)

def convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor, voxel_grid_origin, voxel_size, ply_filename_out, offset=None, scale=None):
    '''
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    '''

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3)
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)