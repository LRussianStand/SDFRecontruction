# -*- coding: utf-8 -*-
import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import renderer
from torchvision import transforms
import pytorch_ssim
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

class Notify(object):
    """Colorful printing prefix.
    A quick example:
    print(Notify.INFO, YOUR TEXT, Notify.ENDC)
    """

    def __init__(self):
        pass

    def HEADER(cls):
        return str(datetime.now()) + ': \033[95m'


    def INFO(cls):
        return str(datetime.now()) + ': \033[92mI'


    def OKBLUE(cls):
        return str(datetime.now()) + ': \033[94m'


    def WARNING(cls):
        return str(datetime.now()) + ': \033[93mW'


    def FAIL(cls):
        return str(datetime.now()) + ': \033[91mF'


    def BOLD(cls):
        return str(datetime.now()) + ': \033[1mB'

    def UNDERLINE(cls):
        return str(datetime.now()) + ': \033[4mU'
    ENDC = '\033[0m'

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

cuda = True if torch.cuda.is_available() else False
print(cuda)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# positional encoding，将输入向量（默认维度3）的每个元素cft扩展到(multires乘以2)个元素
# 输入：扩展维度数量，输出：扩展函数（输入为[bs,3]） 以及 扩展之后的元素数量
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def grid_construction_cube(grid_res, bounding_box_min, bounding_box_max):
    # Construct the sdf grid for a cube with size 2
    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    cube_left_bound_index = float(grid_res - 1) / 4;
    cube_right_bound_index = float(grid_res - 1) / 4 * 3;
    cube_center = float(grid_res - 1) / 2;

    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                if (i >= cube_left_bound_index and i <= cube_right_bound_index and
                        j >= cube_left_bound_index and j <= cube_right_bound_index and
                        k >= cube_left_bound_index and k <= cube_right_bound_index):
                    grid[i, j, k] = voxel_size * max(abs(i - cube_center), abs(j - cube_center),
                                                     abs(k - cube_center)) - 1;
                else:
                    grid[i, j, k] = math.sqrt(
                        pow(voxel_size * (max(i - cube_right_bound_index, cube_left_bound_index - i, 0)), 2) +
                        pow(voxel_size * (max(j - cube_right_bound_index, cube_left_bound_index - j, 0)), 2) +
                        pow(voxel_size * (max(k - cube_right_bound_index, cube_left_bound_index - k, 0)), 2));
    return grid


def read_sdf(file_path, target_grid_res, target_bounding_box_min, target_bounding_box_max, target_voxel_size):
    with open(file_path) as file:
        line = file.readline()

        # Get grid resolutions
        grid_res = line.split()
        grid_res_x = int(grid_res[0])
        grid_res_y = int(grid_res[1])
        grid_res_z = int(grid_res[2])

        # Get bounding box min
        line = file.readline()
        bounding_box_min = line.split()
        bounding_box_min_x = float(bounding_box_min[0])
        bounding_box_min_y = float(bounding_box_min[1])
        bounding_box_min_z = float(bounding_box_min[2])

        line = file.readline()
        voxel_size = float(line)

        # max bounding box (we need to plus 0.0001 to avoid round error)
        bounding_box_max_x = bounding_box_min_x + voxel_size * (grid_res_x - 1)
        bounding_box_max_y = bounding_box_min_y + voxel_size * (grid_res_y - 1)
        bounding_box_max_z = bounding_box_min_z + voxel_size * (grid_res_z - 1)

        min_bounding_box_min = min(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z)
        # print(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z)
        max_bounding_box_max = max(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z)
        # print(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z)
        max_dist = max(bounding_box_max_x - bounding_box_min_x, bounding_box_max_y - bounding_box_min_y,
                       bounding_box_max_z - bounding_box_min_z)

        # max_dist += 0.1
        max_grid_res = max(grid_res_x, grid_res_y, grid_res_z)

        grid = []
        for i in range(grid_res_x):
            grid.append([])
            for j in range(grid_res_y):
                grid[i].append([])
                for k in range(grid_res_z):
                    # grid_value = float(file.readline())
                    grid[i][j].append(2)
                    # lst.append(grid_value)

        for i in range(grid_res_z):
            for j in range(grid_res_y):
                for k in range(grid_res_x):
                    grid_value = float(file.readline())
                    grid[k][j][i] = grid_value

        grid = Tensor(grid)
        target_grid = Tensor(target_grid_res, target_grid_res, target_grid_res)

        linear_space_x = torch.linspace(0, target_grid_res - 1, target_grid_res)
        linear_space_y = torch.linspace(0, target_grid_res - 1, target_grid_res)
        linear_space_z = torch.linspace(0, target_grid_res - 1, target_grid_res)
        first_loop = linear_space_x.repeat(target_grid_res * target_grid_res, 1).t().contiguous().view(-1).unsqueeze_(1)
        second_loop = linear_space_y.repeat(target_grid_res, target_grid_res).t().contiguous().view(-1).unsqueeze_(1)
        third_loop = linear_space_z.repeat(target_grid_res * target_grid_res).unsqueeze_(1)
        loop = torch.cat((first_loop, second_loop, third_loop), 1).cuda()

        min_x = Tensor([bounding_box_min_x]).repeat(target_grid_res * target_grid_res * target_grid_res, 1)
        min_y = Tensor([bounding_box_min_y]).repeat(target_grid_res * target_grid_res * target_grid_res, 1)
        min_z = Tensor([bounding_box_min_z]).repeat(target_grid_res * target_grid_res * target_grid_res, 1)
        bounding_min_matrix = torch.cat((min_x, min_y, min_z), 1)

        move_to_center_x = Tensor([(max_dist - (bounding_box_max_x - bounding_box_min_x)) / 2]).repeat(
            target_grid_res * target_grid_res * target_grid_res, 1)
        move_to_center_y = Tensor([(max_dist - (bounding_box_max_y - bounding_box_min_y)) / 2]).repeat(
            target_grid_res * target_grid_res * target_grid_res, 1)
        move_to_center_z = Tensor([(max_dist - (bounding_box_max_z - bounding_box_min_z)) / 2]).repeat(
            target_grid_res * target_grid_res * target_grid_res, 1)
        move_to_center_matrix = torch.cat((move_to_center_x, move_to_center_y, move_to_center_z), 1)

        # Get the position of the grid points in the refined grid
        points = bounding_min_matrix + target_voxel_size * max_dist / (
                    target_bounding_box_max - target_bounding_box_min) * loop - move_to_center_matrix
        if points[(points[:, 0] < bounding_box_min_x)].shape[0] != 0:
            points[(points[:, 0] < bounding_box_min_x)] = Tensor(
                [bounding_box_max_x, bounding_box_max_y, bounding_box_max_z]).view(1, 3)
        if points[(points[:, 1] < bounding_box_min_y)].shape[0] != 0:
            points[(points[:, 1] < bounding_box_min_y)] = Tensor(
                [bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1, 3)
        if points[(points[:, 2] < bounding_box_min_z)].shape[0] != 0:
            points[(points[:, 2] < bounding_box_min_z)] = Tensor(
                [bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1, 3)
        if points[(points[:, 0] > bounding_box_max_x)].shape[0] != 0:
            points[(points[:, 0] > bounding_box_max_x)] = Tensor(
                [bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1, 3)
        if points[(points[:, 1] > bounding_box_max_y)].shape[0] != 0:
            points[(points[:, 1] > bounding_box_max_y)] = Tensor(
                [bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1, 3)
        if points[(points[:, 2] > bounding_box_max_z)].shape[0] != 0:
            points[(points[:, 2] > bounding_box_max_z)] = Tensor(
                [bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1, 3)
        voxel_min_point_index_x = torch.floor((points[:, 0].unsqueeze_(1) - min_x) / voxel_size).clamp(
            max=grid_res_x - 2)
        voxel_min_point_index_y = torch.floor((points[:, 1].unsqueeze_(1) - min_y) / voxel_size).clamp(
            max=grid_res_y - 2)
        voxel_min_point_index_z = torch.floor((points[:, 2].unsqueeze_(1) - min_z) / voxel_size).clamp(
            max=grid_res_z - 2)
        voxel_min_point_index = torch.cat((voxel_min_point_index_x, voxel_min_point_index_y, voxel_min_point_index_z),
                                          1)
        voxel_min_point = bounding_min_matrix + voxel_min_point_index * voxel_size

        # Compute the sdf value of the grid points in the refined grid
        target_grid = calculate_sdf_value(grid, points, voxel_min_point, voxel_min_point_index, voxel_size, grid_res_x,
                                          grid_res_y, grid_res_z).view(target_grid_res, target_grid_res,
                                                                       target_grid_res)

        # "shortest path" algorithm to fill the values (for changing from "cuboid" SDF to "cube" SDF)
        # min of the SDF values of the closest points + the distance to these points
        # calculate the max resolution get which areas we need to compute the shortest path
        max_res = max(grid_res_x, grid_res_y, grid_res_z)
        if grid_res_x == max_res:
            min_x = 0
            max_x = target_grid_res - 1
            min_y = math.ceil((target_grid_res - target_grid_res / float(grid_res_x) * grid_res_y) / 2)
            max_y = target_grid_res - min_y - 1
            min_z = math.ceil((target_grid_res - target_grid_res / float(grid_res_x) * grid_res_z) / 2)
            max_z = target_grid_res - min_z - 1
        if grid_res_y == max_res:
            min_x = math.ceil((target_grid_res - target_grid_res / float(grid_res_y) * grid_res_x) / 2)
            max_x = target_grid_res - min_x - 1
            min_y = 0
            max_y = target_grid_res - 1
            min_z = math.ceil((target_grid_res - target_grid_res / float(grid_res_y) * grid_res_z) / 2)
            max_z = target_grid_res - min_z - 1
        if grid_res_z == max_res:
            min_x = math.ceil((target_grid_res - target_grid_res / float(grid_res_z) * grid_res_x) / 2)
            max_x = target_grid_res - min_x - 1
            min_y = math.ceil((target_grid_res - target_grid_res / float(grid_res_z) * grid_res_y) / 2)
            max_y = target_grid_res - min_y - 1
            min_z = 0
            max_z = target_grid_res - 1
        min_x = int(min_x)
        max_x = int(max_x)
        min_y = int(min_y)
        max_y = int(max_y)
        min_z = int(min_z)
        max_z = int(max_z)

        # fill the values
        res = target_grid.shape[0]
        for i in range(res):
            for j in range(res):
                for k in range(res):

                    # fill the values outside both x-axis and y-axis
                    if k < min_x and j < min_y:
                        target_grid[k][j][i] = target_grid[min_x][min_y][i] + math.sqrt(
                            (min_x - k) ** 2 + (min_y - j) ** 2) * voxel_size
                    elif k < min_x and j > max_y:
                        target_grid[k][j][i] = target_grid[min_x][max_y][i] + math.sqrt(
                            (min_x - k) ** 2 + (max_y - j) ** 2) * voxel_size
                    elif k > max_x and j < min_y:
                        target_grid[k][j][i] = target_grid[max_x][min_y][i] + math.sqrt(
                            (max_x - k) ** 2 + (min_y - j) ** 2) * voxel_size
                    elif k > max_x and j > max_y:
                        target_grid[k][j][i] = target_grid[max_x][max_y][i] + math.sqrt(
                            (max_x - k) ** 2 + (max_y - j) ** 2) * voxel_size

                    # fill the values outside both x-axis and z-axis
                    elif k < min_x and i < min_z:
                        target_grid[k][j][i] = target_grid[min_x][j][min_z] + math.sqrt(
                            (min_x - k) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif k < min_x and i > max_z:
                        target_grid[k][j][i] = target_grid[min_x][j][max_z] + math.sqrt(
                            (min_x - k) ** 2 + (max_z - i) ** 2) * voxel_size
                    elif k > max_x and i < min_z:
                        target_grid[k][j][i] = target_grid[max_x][j][min_z] + math.sqrt(
                            (max_x - k) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif k > max_x and i > max_z:
                        target_grid[k][j][i] = target_grid[max_x][j][max_z] + math.sqrt(
                            (max_x - k) ** 2 + (max_z - i) ** 2) * voxel_size

                    # fill the values outside both y-axis and z-axis
                    elif j < min_y and i < min_z:
                        target_grid[k][j][i] = target_grid[k][min_y][min_z] + math.sqrt(
                            (min_y - j) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif j < min_y and i > max_z:
                        target_grid[k][j][i] = target_grid[k][min_y][max_z] + math.sqrt(
                            (min_y - j) ** 2 + (max_z - i) ** 2) * voxel_size
                    elif j > max_y and i < min_z:
                        target_grid[k][j][i] = target_grid[k][max_y][min_z] + math.sqrt(
                            (max_y - j) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif j > max_y and i > max_z:
                        target_grid[k][j][i] = target_grid[k][max_y][max_z] + math.sqrt(
                            (max_y - j) ** 2 + (max_z - i) ** 2) * voxel_size

                    # fill the values outside x-axis
                    elif k < min_x:
                        target_grid[k][j][i] = target_grid[min_x][j][i] + math.sqrt((min_x - k) ** 2) * voxel_size
                    elif k > max_x:
                        target_grid[k][j][i] = target_grid[max_x][j][i] + math.sqrt((max_x - k) ** 2) * voxel_size

                    # fill the values outside y-axis
                    elif j < min_y:
                        target_grid[k][j][i] = target_grid[k][min_y][i] + math.sqrt((min_y - j) ** 2) * voxel_size
                    elif j > max_y:
                        target_grid[k][j][i] = target_grid[k][max_y][i] + math.sqrt((max_y - j) ** 2) * voxel_size

                    # fill the values outside z-axis
                    elif i < min_z:
                        target_grid[k][j][i] = target_grid[k][j][min_z] + math.sqrt((min_z - i) ** 2) * voxel_size
                    elif i > max_z:
                        target_grid[k][j][i] = target_grid[k][j][max_z] + math.sqrt((max_z - i) ** 2) * voxel_size

        return target_grid

def grid_construction_torus(grid_res, bounding_box_min, bounding_box_max):
    # radius of the circle between the two circles
    radius_big = 1.5

    # radius of the small circle
    radius_small = 0.5

    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                x = bounding_box_min + voxel_size * i
                y = bounding_box_min + voxel_size * j
                z = bounding_box_min + voxel_size * k

                grid[i, j, k] = math.sqrt(math.pow((math.sqrt(math.pow(y, 2) + math.pow(z, 2)) - radius_big), 2)
                                          + math.pow(x, 2)) - radius_small;

    return grid


def grid_construction_sphere_big(grid_res, bounding_box_min, bounding_box_max):
    # Construct the sdf grid for a sphere with radius 1.6
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1.6
    if cuda:
        return grid.cuda()
    else:
        return grid


def grid_construction_sphere_small(grid_res, bounding_box_min, bounding_box_max,radius):
    # Construct the sdf grid for a sphere with radius 1
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - radius
    if cuda:
        return grid.cuda()
    else:
        return grid


def get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z):
    # largest index
    n_x = grid_res_x - 1
    n_y = grid_res_y - 1
    n_z = grid_res_z - 1

    # x-axis normal vectors
    X_1 = torch.cat(
        (grid[1:, :, :], (3 * grid[n_x, :, :] - 3 * grid[n_x - 1, :, :] + grid[n_x - 2, :, :]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid[1, :, :] + 3 * grid[0, :, :] + grid[2, :, :]).unsqueeze_(0), grid[:n_x, :, :]), 0)
    grid_normal_x = (X_1 - X_2) / (2 * voxel_size)

    # y-axis normal vectors
    Y_1 = torch.cat(
        (grid[:, 1:, :], (3 * grid[:, n_y, :] - 3 * grid[:, n_y - 1, :] + grid[:, n_y - 2, :]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid[:, 1, :] + 3 * grid[:, 0, :] + grid[:, 2, :]).unsqueeze_(1), grid[:, :n_y, :]), 1)
    grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size)

    # z-axis normal vectors
    Z_1 = torch.cat(
        (grid[:, :, 1:], (3 * grid[:, :, n_z] - 3 * grid[:, :, n_z - 1] + grid[:, :, n_z - 2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid[:, :, 1] + 3 * grid[:, :, 0] + grid[:, :, 2]).unsqueeze_(2), grid[:, :, :n_z]), 2)
    grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size)

    return [grid_normal_x, grid_normal_y, grid_normal_z]


def get_intersection_normal(intersection_grid_normal, intersection_pos, voxel_min_point, voxel_size):
    # Compute parameters
    tx = (intersection_pos[:, :, 0] - voxel_min_point[:, :, 0]) / voxel_size
    ty = (intersection_pos[:, :, 1] - voxel_min_point[:, :, 1]) / voxel_size
    tz = (intersection_pos[:, :, 2] - voxel_min_point[:, :, 2]) / voxel_size

    intersection_normal = (1 - tz) * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 0] \
                          + tz * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 1] \
                          + (1 - tz) * ty * (1 - tx) * intersection_grid_normal[:, :, 2] \
                          + tz * ty * (1 - tx) * intersection_grid_normal[:, :, 3] \
                          + (1 - tz) * (1 - ty) * tx * intersection_grid_normal[:, :, 4] \
                          + tz * (1 - ty) * tx * intersection_grid_normal[:, :, 5] \
                          + (1 - tz) * ty * tx * intersection_grid_normal[:, :, 6] \
                          + tz * ty * tx * intersection_grid_normal[:, :, 7]

    return intersection_normal

def get_intersection_normal_bs(intersection_grid_normal, intersection_pos, voxel_min_point, voxel_size):
    # Compute parameters
    tx = (intersection_pos[:, :, :, 0] - voxel_min_point[:, :, :, 0]) / voxel_size
    ty = (intersection_pos[:, :, :, 1] - voxel_min_point[:, :, :, 1]) / voxel_size
    tz = (intersection_pos[:, :, :, 2] - voxel_min_point[:, :, :, 2]) / voxel_size

    intersection_normal = (1 - tz) * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, :, 0] \
                          + tz * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, :, 1] \
                          + (1 - tz) * ty * (1 - tx) * intersection_grid_normal[:, :, :, 2] \
                          + tz * ty * (1 - tx) * intersection_grid_normal[:, :, :, 3] \
                          + (1 - tz) * (1 - ty) * tx * intersection_grid_normal[:, :, :, 4] \
                          + tz * (1 - ty) * tx * intersection_grid_normal[:, :, :, 5] \
                          + (1 - tz) * ty * tx * intersection_grid_normal[:, :, :, 6] \
                          + tz * ty * tx * intersection_grid_normal[:, :, :, 7]

    return intersection_normal


# Do one more step for ray matching
def calculate_sdf_value(grid, points, voxel_min_point, voxel_min_point_index, voxel_size, grid_res_x, grid_res_y,
                        grid_res_z):
    string = ""

    # Linear interpolate along x axis the eight values
    tx = (points[:, 0] - voxel_min_point[:, 0]) / voxel_size;
    string = string + "\n\nvoxel_size: \n" + str(voxel_size)
    string = string + "\n\ntx: \n" + str(tx)
    print(grid.shape)

    if cuda:
        tx = tx.cuda()
        x = voxel_min_point_index.long()[:, 0]
        y = voxel_min_point_index.long()[:, 1]
        z = voxel_min_point_index.long()[:, 2]

        string = string + "\n\nx: \n" + str(x)
        string = string + "\n\ny: \n" + str(y)
        string = string + "\n\nz: \n" + str(z)

        c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z];
        c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z];
        c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1];
        c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1, z + 1];

        string = string + "\n\n(1 - tx): \n" + str((1 - tx))
        string = string + "\n\ngrid[x,y,z]: \n" + str(grid[x, y, z])
        string = string + "\n\ngrid[x+1,y,z]: \n" + str(grid[x + 1, y, z])
        string = string + "\n\nc01: \n" + str(c01)
        string = string + "\n\nc23: \n" + str(c23)
        string = string + "\n\nc45: \n" + str(c45)
        string = string + "\n\nc67: \n" + str(c67)

        # Linear interpolate along the y axis
        ty = (points[:, 1] - voxel_min_point[:, 1]) / voxel_size;
        ty = ty.cuda()
        c0 = (1 - ty) * c01 + ty * c23;
        c1 = (1 - ty) * c45 + ty * c67;

        string = string + "\n\nty: \n" + str(ty)

        string = string + "\n\nc0: \n" + str(c0)
        string = string + "\n\nc1: \n" + str(c1)

        # Return final value interpolated along z
        tz = (points[:, 2] - voxel_min_point[:, 2]) / voxel_size;
        tz = tz.cuda()
        string = string + "\n\ntz: \n" + str(tz)

    else:
        x = voxel_min_point_index.numpy()[:, 0]
        y = voxel_min_point_index.numpy()[:, 1]
        z = voxel_min_point_index.numpy()[:, 2]

        c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z];
        c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z];
        c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1];
        c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1, z + 1];

        # Linear interpolate along the y axis
        ty = (points[:, 1] - voxel_min_point[:, 1]) / voxel_size;
        c0 = (1 - ty) * c01 + ty * c23;
        c1 = (1 - ty) * c45 + ty * c67;

        # Return final value interpolated along z
        tz = (points[:, 2] - voxel_min_point[:, 2]) / voxel_size;

    result = (1 - tz) * c0 + tz * c1;

    return result

def compute_intersection_pos(grid, intersection_pos_rough, voxel_min_point, voxel_min_point_index, ray_direction,
                             voxel_size, mask, width, height):
    # Linear interpolate along x axis the eight values
    tx = (intersection_pos_rough[ :, :, 0] - voxel_min_point[ :, :, 0]) / voxel_size

    if cuda:

        x = voxel_min_point_index.long()[ :, :, 0]
        y = voxel_min_point_index.long()[ :, :, 1]
        z = voxel_min_point_index.long()[ :, :, 2]

        c01 = (1 - tx) * grid[x, y, z].cuda() + tx * grid[x + 1, y, z].cuda()
        c23 = (1 - tx) * grid[x, y + 1, z].cuda() + tx * grid[x + 1, y + 1, z].cuda()
        c45 = (1 - tx) * grid[x, y, z + 1].cuda() + tx * grid[x + 1, y, z + 1].cuda()
        c67 = (1 - tx) * grid[x, y + 1, z + 1].cuda() + tx * grid[x + 1, y + 1, z + 1].cuda()

    else:
        x = voxel_min_point_index.numpy()[ :, :, 0]
        y = voxel_min_point_index.numpy()[ :, :, 1]
        z = voxel_min_point_index.numpy()[ :, :, 2]

        c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z]
        c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z]
        c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1]
        c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1, z + 1]

        # Linear interpolate along the y axis
    ty = (intersection_pos_rough[ :, :, 1] - voxel_min_point[ :, :, 1]) / voxel_size
    c0 = (1 - ty) * c01 + ty * c23
    c1 = (1 - ty) * c45 + ty * c67

    # Return final value interpolated along z
    tz = (intersection_pos_rough[ :, :, 2] - voxel_min_point[:, :, 2]) / voxel_size

    sdf_value = (1 - tz) * c0 + tz * c1

    return (intersection_pos_rough + ray_direction * sdf_value.view( height, width, 1).repeat( 1, 1, 3)) + (1 - mask.view( height, width, 1).repeat( 1, 1, 3))

def compute_intersection_pos_bs(grid, intersection_pos_rough, voxel_min_point, voxel_min_point_index, ray_direction,
                             voxel_size, mask,batchsize, width, height):
    # Linear interpolate along x axis the eight values
    tx = (intersection_pos_rough[:, :, :, 0] - voxel_min_point[:, :, :, 0]) / voxel_size

    if cuda:

        x = voxel_min_point_index.long()[:, :, :, 0]
        y = voxel_min_point_index.long()[:, :, :, 1]
        z = voxel_min_point_index.long()[:, :, :, 2]

        c01 = (1 - tx) * grid[x, y, z].cuda() + tx * grid[x + 1, y, z].cuda()
        c23 = (1 - tx) * grid[x, y + 1, z].cuda() + tx * grid[x + 1, y + 1, z].cuda()
        c45 = (1 - tx) * grid[x, y, z + 1].cuda() + tx * grid[x + 1, y, z + 1].cuda()
        c67 = (1 - tx) * grid[x, y + 1, z + 1].cuda() + tx * grid[x + 1, y + 1, z + 1].cuda()

    else:
        x = voxel_min_point_index.numpy()[:, :, :, 0]
        y = voxel_min_point_index.numpy()[:, :, :, 1]
        z = voxel_min_point_index.numpy()[:, :, :, 2]

        c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z]
        c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z]
        c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1]
        c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1, z + 1]

        # Linear interpolate along the y axis
    ty = (intersection_pos_rough[:, :, :, 1] - voxel_min_point[:, :, :, 1]) / voxel_size
    c0 = (1 - ty) * c01 + ty * c23
    c1 = (1 - ty) * c45 + ty * c67

    # Return final value interpolated along z
    tz = (intersection_pos_rough[:, :, :, 2] - voxel_min_point[:, :, :, 2]) / voxel_size

    sdf_value = (1 - tz) * c0 + tz * c1

    return (intersection_pos_rough + ray_direction * sdf_value.view(batchsize, height, width, 1).repeat(1, 1, 1, 3)) \
           + (1 - mask.view(batchsize, height, width, 1).repeat(1, 1, 1, 3))


def generate_image2(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                   bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                   voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid, fov, origin, lookat, up, eta1,
                   eta2,envmap,envHeight,envWidth):
    # Get normal vectors for points on the grid
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)
    w_h_3 = torch.zeros(height, width, 3).cuda()
    w_h = torch.zeros(height, width).cuda()

    # Generate rays
    # Do ray tracing in cpp
    outputs = renderer.ray_matching_cam(w_h_3, w_h, grid, width, height, bounding_box_min_x, bounding_box_min_y,
                                        bounding_box_min_z, \
                                        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                                        grid_res_x, grid_res_y, grid_res_z, fov, origin, lookat, up
                                        )

    # {intersection_pos, voxel_position, directions}
    ray_direction = outputs[0]
    intersection_pos_rough = outputs[1]
    voxel_min_point_index = outputs[2]
    origin_image_distances = outputs[3]
    pixel_distances = outputs[4]
    l_t2 = outputs[5]
    l_r = outputs[6]
    attenuate1 = outputs[7].unsqueeze(2)
    attenuate2 = outputs[8].unsqueeze(2)


    mask = (voxel_min_point_index[:, :, 0] != -1).type(Tensor).unsqueeze(2)

    l_r = l_r * mask
    attenuate1 = attenuate1 * mask
    attenuate2 = attenuate2 * mask
    l_t2 = ray_direction * (1 - mask) + l_t2 * mask


    #render image!!!
    refractImg = sampleEnvLight(l_t2, envmap,envHeight,envWidth,height,width)
    refractImg = refractImg * (1 - attenuate1) * (1 - attenuate2)


    reflectImg = sampleEnvLight(l_r, envmap,envHeight,envWidth,height,width)
    reflectImg = reflectImg * attenuate1
    #reflectImg = reflectImg * 0

    return refractImg, reflectImg, mask

#-----------------------------------------------------------------------------------------------------------------

def generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                   bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                   voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid, fov, origin, lookat, up, eta1,
                   eta2,envmap,envHeight,envWidth):
    #测试代码，更改相机参数
    # origin = Tensor([[0,0,-2]])
    # lookat = Tensor([[0,0,0]])
    # up = Tensor([[0,1,0]])
    origin = origin.unsqueeze(0)
    lookat=lookat.unsqueeze(0)
    up=up.unsqueeze(0)
    b_w_h_3 = torch.zeros(1,height, width, 3).cuda()
    b_w_h = torch.zeros(1, height, width).cuda()
    envmap = envmap.unsqueeze(0)

    # Get normal vectors for points on the grid
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)
    w_h_3 = torch.zeros(height, width, 3).cuda()
    w_h = torch.zeros(height, width).cuda()

    # Generate rays
    # Do ray tracing in cpp
    outputs = renderer.bs_ray_matching_cam(b_w_h_3, b_w_h, grid, 1, width, height, bounding_box_min_x, bounding_box_min_y,
                                        bounding_box_min_z, \
                                        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                                        grid_res_x, grid_res_y, grid_res_z, fov, origin, lookat, up
                                        )

    # {intersection_pos, voxel_position, directions}
    ray_direction = outputs[0]
    intersection_pos_rough = outputs[1]
    voxel_min_point_index = outputs[2]
    origin_image_distances = outputs[3]
    pixel_distances = outputs[4]
    watch = ray_direction.cpu().numpy()
    mask = (voxel_min_point_index[:, :, :, 0] != -1).type(Tensor).unsqueeze(3)

    # get fine pos and normal
    intersection_pos1, intersection_normal1 = get_fine_pos_and_normal_bs(1,width, height, bounding_box_min_x,
                                                                      bounding_box_min_y, bounding_box_min_z,
                                                                      voxel_size, grid, grid_res_x, grid_res_y,
                                                                      grid_res_z, grid_normal_x, grid_normal_y,
                                                                      grid_normal_z, ray_direction,
                                                                      intersection_pos_rough, voxel_min_point_index)

    # get refraction, reflection, attenuation for the front surface
    l_t, attenuate1, mask1 = refraction_bs(ray_direction, intersection_normal1, eta1, eta2)
    l_r = reflection_bs(ray_direction, intersection_normal1)

    l_t = ray_direction * (1 - mask) + l_t * mask
    l_r = l_r * mask
    attenuate1 = attenuate1 * mask
    mask1 = mask * mask1

    # get rough intersect pos for the back-surface
    # dis2box_xyz = torch.abs(
    #     ((ray_direction > 0).type(Tensor) * Tensor([bounding_box_max_x, bounding_box_max_y, bounding_box_max_z]) + (
    #                 ray_direction < 0).type(Tensor) * Tensor(
    #         [bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) - intersection_pos1) / ray_direction)
    # dis2box_xyz = torch.min(dis2box_xyz,2,keepdim=True)[0]

    dis2box_xyz = torch.ones_like(mask) * math.sqrt(3) * (bounding_box_max_x - bounding_box_min_x)
    origin2 = intersection_pos1 + l_t * dis2box_xyz
    outputs2 = renderer.bs_ray_matching_dir(b_w_h_3, b_w_h, grid,1, width, height, bounding_box_min_x, bounding_box_min_y,
                                         bounding_box_min_z, bounding_box_max_x, bounding_box_max_y, bounding_box_max_z,
                                         grid_res_x, grid_res_y, grid_res_z, origin2,-l_t,origin_image_distances,pixel_distances)
    intersection_pos_rough2 = outputs2[0]
    voxel_min_point_index2 = outputs2[1]

    # get fine pos and normal for the back-surface
    intersection_pos2, intersection_normal2 = get_fine_pos_and_normal_bs(1, width, height, bounding_box_min_x,
                                                                      bounding_box_min_y, bounding_box_min_z,
                                                                      voxel_size, grid, grid_res_x, grid_res_y,
                                                                      grid_res_z, grid_normal_x, grid_normal_y,
                                                                      grid_normal_z, -l_t,
                                                                      intersection_pos_rough2, voxel_min_point_index2)

    # get refraction, reflection, attenuation for the back surface
    l_t2, attenuate2, mask2 = refraction_bs(l_t, -intersection_normal2, eta2, eta1)
    l_r2 = reflection_bs(l_t, -intersection_normal2)

    l_t2 = ray_direction * (1 - mask) + l_t2 * mask
    l_r2 = l_r2 * mask
    attenuate2 = attenuate2 * mask
    mask2 = mask * mask2

    #render image!!!
    refractImg = sampleEnvLight_bs(l_t2, envmap,envHeight,envWidth,1, height,width)
    refractImg = refractImg * (1 - attenuate1) * (1 - attenuate2)
    #refractImg = refractImg * 0

    reflectImg = sampleEnvLight_bs(l_r, envmap,envHeight,envWidth,1, height,width)
    reflectImg = reflectImg * attenuate1
    #reflectImg = reflectImg * 0

    maskatt = torch.clamp((mask1 + mask2).float(), 0, 1)

    return refractImg, reflectImg, maskatt, mask

def generate_image_bs(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                   bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                   voxel_size, grid_res_x, grid_res_y, grid_res_z, batchsize, width, height, grid, fov, origin, lookat, up, eta1,
                   eta2,envmap,envHeight,envWidth):
    b_w_h_3 = torch.zeros(batchsize,height, width, 3).cuda()
    b_w_h = torch.zeros(batchsize, height, width).cuda()
    err_code =  - 7

    # Get normal vectors for points on the grid
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)

    # Generate rays
    # Do ray tracing in cpp
    outputs = renderer.bs_ray_matching_cam(b_w_h_3, b_w_h, grid, batchsize, width, height, bounding_box_min_x, bounding_box_min_y,
                                        bounding_box_min_z, \
                                        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                                        grid_res_x, grid_res_y, grid_res_z, fov, origin, lookat, up, err_code
                                        )

    # {intersection_pos, voxel_position, directions}
    ray_direction = outputs[0]
    intersection_pos_rough = outputs[1]
    voxel_min_point_index = outputs[2]
    origin_image_distances = outputs[3]
    pixel_distances = outputs[4]
    mask = (voxel_min_point_index[:, :, :, 0] != err_code).type(Tensor).unsqueeze(3)

    # get fine pos and normal
    intersection_pos1, intersection_normal1 = get_fine_pos_and_normal_bs(batchsize,width, height, bounding_box_min_x,
                                                                      bounding_box_min_y, bounding_box_min_z,
                                                                      voxel_size, grid, grid_res_x, grid_res_y,
                                                                      grid_res_z, grid_normal_x, grid_normal_y,
                                                                      grid_normal_z, ray_direction,
                                                                      intersection_pos_rough, voxel_min_point_index, err_code)

    # get refraction, reflection, attenuation for the front surface
    l_t, attenuate1, mask1 = refraction_bs(ray_direction, intersection_normal1, eta1, eta2)
    l_r = reflection_bs(ray_direction, intersection_normal1)

    l_t = ray_direction * (1 - mask) + l_t * mask
    l_r = l_r * mask + ray_direction * (1 - mask)
    attenuate1 = attenuate1 * mask
    mask1 = mask * mask1

    # get rough intersect pos for the back-surface
    # dis2box_xyz = torch.abs(
    #     ((ray_direction > 0).type(Tensor) * Tensor([bounding_box_max_x, bounding_box_max_y, bounding_box_max_z]) + (
    #                 ray_direction < 0).type(Tensor) * Tensor(
    #         [bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) - intersection_pos1) / ray_direction)
    # dis2box_xyz = torch.min(dis2box_xyz,2,keepdim=True)[0]


    #------------------------------------------------------------------------------------------------------------------------
    dis2box_xyz = torch.ones_like(mask) * math.sqrt(3) * (bounding_box_max_x - bounding_box_min_x)
    origin2 = intersection_pos1 + l_t * dis2box_xyz
    outputs2 = renderer.bs_ray_matching_dir(b_w_h_3, b_w_h, grid,batchsize, width, height, bounding_box_min_x, bounding_box_min_y,
                                         bounding_box_min_z, bounding_box_max_x, bounding_box_max_y, bounding_box_max_z,
                                         grid_res_x, grid_res_y, grid_res_z, origin2,-l_t,origin_image_distances,pixel_distances, err_code)
    intersection_pos_rough2 = outputs2[0]
    voxel_min_point_index2 = outputs2[1]

    # get fine pos and normal for the back-surface
    intersection_pos2, intersection_normal2 = get_fine_pos_and_normal_bs(batchsize, width, height, bounding_box_min_x,
                                                                      bounding_box_min_y, bounding_box_min_z,
                                                                      voxel_size, grid, grid_res_x, grid_res_y,
                                                                      grid_res_z, grid_normal_x, grid_normal_y,
                                                                      grid_normal_z, -l_t,
                                                                      intersection_pos_rough2, voxel_min_point_index2, err_code)

    # get refraction, reflection, attenuation for the back surface
    l_t2, attenuate2, mask2 = refraction_bs(l_t, -intersection_normal2, eta2, eta1)
    l_r2 = reflection_bs(l_t, -intersection_normal2)

    l_t2 = ray_direction * (1 - mask) + l_t2 * mask
    l_r2 = l_r2 * mask + ray_direction * (1 - mask)
    attenuate2 = attenuate2 * mask
    mask2 = mask * mask2


    #----------------------------------------------------------------------------------------------------------------------
    origin3 = intersection_pos2 + l_r2 * dis2box_xyz
    outputs3 = renderer.bs_ray_matching_dir(b_w_h_3, b_w_h, grid, batchsize, width, height, bounding_box_min_x,
                                            bounding_box_min_y,
                                            bounding_box_min_z, bounding_box_max_x, bounding_box_max_y,
                                            bounding_box_max_z,
                                            grid_res_x, grid_res_y, grid_res_z, origin3, -l_r2, origin_image_distances,
                                            pixel_distances, err_code)
    intersection_pos_rough3 = outputs3[0]
    voxel_min_point_index3 = outputs3[1]
    intersection_pos3, intersection_normal3 = get_fine_pos_and_normal_bs(batchsize, width, height, bounding_box_min_x,
                                                                         bounding_box_min_y, bounding_box_min_z,
                                                                         voxel_size, grid, grid_res_x, grid_res_y,
                                                                         grid_res_z, grid_normal_x, grid_normal_y,
                                                                         grid_normal_z, -l_r2,
                                                                         intersection_pos_rough3,
                                                                         voxel_min_point_index3, err_code)
    #get refraction only
    l_t3, attenuate3, mask3 = refraction_bs(l_r2, -intersection_normal3, eta2, eta1)
    l_r3 = reflection_bs(l_r2, -intersection_normal3)
    attenuate3 = attenuate3 * mask
    mask3 = mask * mask3
    l_r3 = l_r3 * mask + ray_direction * (1 - mask)

    #-----------------------------------------------------------------------------------------------------------------------
    origin4 = intersection_pos3 + l_r3 * dis2box_xyz
    outputs4 = renderer.bs_ray_matching_dir(b_w_h_3, b_w_h, grid, batchsize, width, height, bounding_box_min_x,
                                            bounding_box_min_y,
                                            bounding_box_min_z, bounding_box_max_x, bounding_box_max_y,
                                            bounding_box_max_z,
                                            grid_res_x, grid_res_y, grid_res_z, origin4, -l_r3, origin_image_distances,
                                            pixel_distances, err_code)
    intersection_pos_rough4 = outputs4[0]
    voxel_min_point_index4 = outputs4[1]
    intersection_pos4, intersection_normal4 = get_fine_pos_and_normal_bs(batchsize, width, height, bounding_box_min_x,
                                                                         bounding_box_min_y, bounding_box_min_z,
                                                                         voxel_size, grid, grid_res_x, grid_res_y,
                                                                         grid_res_z, grid_normal_x, grid_normal_y,
                                                                         grid_normal_z, -l_r3,
                                                                         intersection_pos_rough4,
                                                                         voxel_min_point_index4, err_code)
    # get refraction only
    l_t4, attenuate4, mask4 = refraction_bs(l_r3, -intersection_normal4, eta2, eta1)
    l_r4 = reflection_bs(l_r3, -intersection_normal4)
    attenuate4 = attenuate4 * mask
    mask4 = mask * mask3
    l_r4 = l_r4 * mask + ray_direction * (1 - mask)

    #-----------------------------------------------------------------------------------------------------------------------
    #render image!!!
    refractImg1 = sampleEnvLight_bs(l_t2, envmap,envHeight,envWidth,batchsize, height,width)
    refractImg1 = refractImg1 * (1 - attenuate1) * (1 - attenuate2)


    reflectImg = sampleEnvLight_bs(l_r, envmap,envHeight,envWidth,batchsize, height,width)
    reflectImg = reflectImg * attenuate1 * mask

    refractImg2 = sampleEnvLight_bs(l_t3, envmap,envHeight,envWidth,batchsize, height,width)
    refractImg2 = refractImg2 * (1 - attenuate1) * attenuate2 * (1 - attenuate3) * mask

    refractImg3 = sampleEnvLight_bs(l_t4, envmap, envHeight, envWidth, batchsize, height, width)
    refractImg3 = refractImg3 * (1 - attenuate1) * attenuate2 * (attenuate3) * (1 - attenuate4) * mask

    maskatt = torch.clamp((mask1 + mask2 + mask3 + mask4).float(), 0, 1)

    relative_base = torch.floor((intersection_pos1 - bounding_box_min_x)/voxel_size).data


    return (refractImg1 + reflectImg + refractImg2 + refractImg3),maskatt, mask, voxel_min_point_index,intersection_pos1

def generate_image_lam(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                   bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                   voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid,  fov, origin, lookat, up):
    # Get normal vectors for points on the grid
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)


    w_h_3 = torch.zeros(height, width, 3).cuda()
    w_h = torch.zeros(height, width).cuda()


    # Do ray tracing in cpp
    outputs = renderer.ray_matching_cam(w_h_3, w_h, grid, width, height, bounding_box_min_x, bounding_box_min_y,
                                        bounding_box_min_z, \
                                        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                                        grid_res_x, grid_res_y, grid_res_z, fov, origin, lookat, up
                                        )

    # {intersection_pos, voxel_position, directions}
    intersection_pos_rough = outputs[1]
    voxel_min_point_index = outputs[2]
    ray_direction = outputs[0]

    # Initialize grid values and normals for intersection voxels
    intersection_grid_normal_x = Tensor(width, height, 8)
    intersection_grid_normal_y = Tensor(width, height, 8)
    intersection_grid_normal_z = Tensor(width, height, 8)
    intersection_grid = Tensor(width, height, 8)

    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:, :, 0] != -1).type(Tensor)

    # Get the indices of the minimum point of the intersecting voxels
    x = voxel_min_point_index[:, :, 0].type(torch.cuda.LongTensor)
    y = voxel_min_point_index[:, :, 1].type(torch.cuda.LongTensor)
    z = voxel_min_point_index[:, :, 2].type(torch.cuda.LongTensor)
    x[x == -1] = 0
    y[y == -1] = 0
    z[z == -1] = 0

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    x1 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    x2 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    x3 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    x4 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(2)
    x5 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(2)
    x6 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    x7 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    x8 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2) + (
                1 - mask.view(height, width, 1).repeat(1, 1, 8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    y2 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    y3 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    y4 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(2)
    y5 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(2)
    y6 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    y7 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    y8 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 2) + (
                1 - mask.view(height, width, 1).repeat(1, 1, 8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    z2 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    z3 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    z4 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(2)
    z5 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(2)
    z6 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    z7 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    z8 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_z = torch.cat((z1, z2, z3, z4, z5, z6, z7, z8), 2) + (
                1 - mask.view(height, width, 1).repeat(1, 1, 8))

    # Change from grid coordinates to world coordinates
    voxel_min_point = Tensor(
        [bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos(grid, intersection_pos_rough,
                                                voxel_min_point, voxel_min_point_index,
                                                ray_direction, voxel_size, mask,width,height)

    intersection_pos = intersection_pos * mask.repeat(3, 1, 1).permute(1, 2, 0)
    shading = Tensor(height, width).fill_(0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal(intersection_grid_normal_x, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_y = get_intersection_normal(intersection_grid_normal_y, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_z = get_intersection_normal(intersection_grid_normal_z, intersection_pos, voxel_min_point,
                                                    voxel_size)

    # Put all the xyz-axis of the normal vectors into a single matrix
    intersection_normal_x_resize = intersection_normal_x.unsqueeze_(2)
    intersection_normal_y_resize = intersection_normal_y.unsqueeze_(2)
    intersection_normal_z_resize = intersection_normal_z.unsqueeze_(2)
    intersection_normal = torch.cat(
        (intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 2)
    intersection_normal = intersection_normal / torch.unsqueeze(torch.norm(intersection_normal, p=2, dim=2), 2).repeat(
        1, 1, 3)

    # Create the point light
    light_position = origin.repeat(height, width, 1)
    light_norm = torch.unsqueeze(torch.norm(light_position - intersection_pos, p=2, dim=2), 2).repeat(1, 1, 3)
    light_direction_point = (light_position - intersection_pos) / light_norm

    # Create the directional light
    shading = 0
    light_direction = (origin / torch.norm(origin, p=2)).repeat(height, width, 1)
    l_dot_n = torch.sum(light_direction * intersection_normal, 2).unsqueeze_(2)
    shading += 10 * torch.max(l_dot_n, Tensor(height, width, 1).fill_(0))[:, :, 0] / torch.pow(
        torch.sum((light_position - intersection_pos) * light_direction_point, dim=2), 2)

    # Get the final image
    image = shading * mask
    image[mask == 0] = 0

    return image
# The energy E captures the difference between a rendered image and
# a desired target image, and the rendered image is a function of the
# SDF values. You could write E(SDF) = ||rendering(SDF)-target_image||^2.
# In addition, there is a second term in the energy as you observed that
# constrains the length of the normal of the SDF to 1. This is a regularization
# term to make sure the output is still a valid SDF.
def loss_fn(output, target, grid, voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height):
    image_loss = torch.sum(torch.abs(target - output))  # / (width * height)

    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)
    sdf_loss = torch.sum(torch.abs(torch.pow(grid_normal_x[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + torch.pow(grid_normal_y[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + torch.pow(grid_normal_z[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1],
                                               2) - 1))  # / ((grid_res-1) * (grid_res-1) * (grid_res-1))

    print("\n\nimage loss: ", image_loss)
    print("sdf loss: ", sdf_loss)

    return image_loss, sdf_loss



def loss_fn_ssim(output, target, grid, voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height):
    def RGB2GREY(tensor):
        R = tensor[0]
        G = tensor[1]
        B = tensor[2]
        grey = 0.299 * R + 0.587 * G + 0.114 * B
        grey = grey.view(1, height, width)
        return grey

    output = RGB2GREY(output).unsqueeze(0)
    targetg = RGB2GREY(target).unsqueeze(0)
    ssim = pytorch_ssim.SSIM()

    image_loss = (1 - ssim(output,targetg))

    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)
    sdf_loss = torch.sum(torch.abs(torch.pow(grid_normal_x[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + torch.pow(grid_normal_y[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + torch.pow(grid_normal_z[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1],
                                               2) - 1))   / ((grid_res_x-1) * (grid_res_y-1) * (grid_res_z-1))

    print("\n\nimage loss: ", image_loss)
    print("sdf loss: ", sdf_loss)

    return image_loss, sdf_loss

def loss_fn_ssim_bs(output, target, grid, voxel_size, grid_res_x, grid_res_y, grid_res_z, batchsize, width, height):
    def RGB2GREY(tensor):
        R = tensor[:,0,:,:]
        G = tensor[:,1,:,:]
        B = tensor[:,2,:,:]
        grey = 0.299 * R + 0.587 * G + 0.114 * B
        grey = grey.view(batchsize, 1,height, width)
        return grey

    outputg = RGB2GREY(output)
    targetg = RGB2GREY(target)
    ssim = pytorch_ssim.SSIM()

    image_loss = (1 - ssim(outputg,targetg))
    #image_loss = torch.sum(torch.abs(target - output)) /(batchsize*width*height)

    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)
    sdf_loss = torch.sum(torch.abs(torch.pow(grid_normal_x[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + torch.pow(grid_normal_y[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + torch.pow(grid_normal_z[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1],
                                               2) - 1))   / ((grid_res_x-1) * (grid_res_y-1) * (grid_res_z-1))

    print("\n\nimage loss: ", image_loss.item())
    print("sdf loss: ", sdf_loss.item())

    return image_loss, sdf_loss

def loss_mask_chamfer_bs(mask_gt,mask_pred,fine_pos,cameras):
    """the fine_pos (n,h,w,3) must not be masked!!!
        as the number of the mask_gt points are different, the chamfer distance calculation for each input image will be seperated!
    """
    camNum = mask_gt.size(0)
    height = mask_gt.size(1)
    width = mask_gt.size(2)
    proj_points = cameras.transform_points_screen(points=fine_pos.reshape(camNum, -1, 3),
                                    image_size=np.tile(np.array((width, height)),(camNum, 1)))
    list_proj_points = torch.split(proj_points.reshape(camNum,height,width,3),1,dim=0)

    list_mask_gt = torch.split(mask_gt.squeeze(-1),1,dim=0)
    list_mask_pred = torch.split(mask_pred.squeeze(-1),1,dim=0)

    losses = []
    for cam in range(len(list_mask_gt)):
        points_gt = torch.nonzero(list_mask_gt[cam].squeeze(0),as_tuple=False)   #(n_gt,2[row,col])
        mask_pred_row,mask_pred_col = torch.nonzero(list_mask_pred[cam].squeeze(0),as_tuple=True)   #(n_pred,2[row,col])
        points_proj = list_proj_points[cam].squeeze(0)[mask_pred_row,mask_pred_col]
        points_proj_row = points_proj[:,1]
        points_proj_col = width - 1 - points_proj[:,0]
        points_proj = torch.stack((points_proj_row,points_proj_col),dim=-1)
        loss,_ = chamfer_distance(points_gt.unsqueeze(0).float(),points_proj.unsqueeze(0),point_reduction = "mean")
        losses.append(loss)
    return torch.stack(losses).mean()

def loss_mask_chamfer_bs2(mask_gt,mask_pred,fine_pos,cameras):
    """the fine_pos (n,h,w,3) must not be masked!!!
        as the number of the mask_gt points are different, the chamfer distance calculation for each input image will be seperated!
    """
    camNum = mask_gt.size(0)
    height = mask_gt.size(1)
    width = mask_gt.size(2)
    proj_points = cameras.transform_points_screen(points=fine_pos.reshape(camNum, -1, 3),
                                    image_size=np.tile(np.array((width, height)),(camNum, 1))).reshape(camNum,height,width,3)
    points_gt = torch.nonzero(mask_gt, as_tuple=False)  #(n_pred,2[row,col])






def get_fine_pos_and_normal(width, height, bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, voxel_size, grid,
                            grid_res_x, grid_res_y, grid_res_z, grid_normal_x, grid_normal_y, grid_normal_z,
                            ray_direction, intersection_pos_rough, voxel_min_point_index):
    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:, :, 0] != -1).type(Tensor)

    # Get the indices of the minimum point of the intersecting voxels
    x = voxel_min_point_index[:, :, 0].type(torch.cuda.LongTensor)
    y = voxel_min_point_index[:, :, 1].type(torch.cuda.LongTensor)
    z = voxel_min_point_index[:, :, 2].type(torch.cuda.LongTensor)
    x[x == -1] = 0
    y[y == -1] = 0
    z[z == -1] = 0

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    x1 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    x2 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    x3 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    x4 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(2)
    x5 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(2)
    x6 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    x7 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    x8 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2) + (
            1 - mask.view(height, width, 1).repeat(1, 1, 8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    y2 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    y3 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    y4 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(2)
    y5 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(2)
    y6 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    y7 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    y8 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 2) + (
            1 - mask.view(height, width, 1).repeat(1, 1, 8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    z2 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    z3 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(2)
    z4 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(2)
    z5 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(2)
    z6 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    z7 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    z8 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_z = torch.cat((z1, z2, z3, z4, z5, z6, z7, z8), 2) + (
            1 - mask.view(height, width, 1).repeat(1, 1, 8))

    # Change from grid coordinates to world coordinates
    voxel_min_point = Tensor(
        [bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos(grid, intersection_pos_rough,
                                                voxel_min_point, voxel_min_point_index,
                                                ray_direction, voxel_size, mask,width,height)

    intersection_pos = intersection_pos * mask.repeat(3, 1, 1).permute(1, 2, 0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal(intersection_grid_normal_x, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_y = get_intersection_normal(intersection_grid_normal_y, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_z = get_intersection_normal(intersection_grid_normal_z, intersection_pos, voxel_min_point,
                                                    voxel_size)

    # Put all the xyz-axis of the normal vectors into a single matrix
    intersection_normal_x_resize = intersection_normal_x.unsqueeze_(2)
    intersection_normal_y_resize = intersection_normal_y.unsqueeze_(2)
    intersection_normal_z_resize = intersection_normal_z.unsqueeze_(2)
    intersection_normal = torch.cat(
        (intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 2)
    intersection_normal = intersection_normal / torch.unsqueeze(torch.norm(intersection_normal, p=2, dim=2), 2).repeat(
        1, 1, 3)

    return intersection_pos, intersection_normal

#多batchsize版本
def get_fine_pos_and_normal_bs(batchsize, width, height, bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, voxel_size, grid,
                            grid_res_x, grid_res_y, grid_res_z, grid_normal_x, grid_normal_y, grid_normal_z,
                            ray_direction, intersection_pos_rough, voxel_min_point_index, error_code):
    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:, :, :, 0] != error_code).type(Tensor)

    # Get the indices of the minimum point of the intersecting voxels
    x = voxel_min_point_index[:, :, :, 0].type(torch.cuda.LongTensor)
    y = voxel_min_point_index[:, :, :, 1].type(torch.cuda.LongTensor)
    z = voxel_min_point_index[:, :, :, 2].type(torch.cuda.LongTensor)
    x[x == error_code] = 0
    y[y == error_code] = 0
    z[z == error_code] = 0
    x = x.clamp(0,grid_res_x - 2)
    y = y.clamp(0, grid_res_y - 2)
    z = z.clamp(0, grid_res_z - 2)

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    x1 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    x2 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    x3 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    x4 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(3)
    x5 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(3)
    x6 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    x7 = torch.index_select(grid_normal_x.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    x8 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    intersection_grid_normal_x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 3) + (
            1 - mask.view(batchsize,height, width, 1).repeat(1,1, 1, 8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    y2 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    y3 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    y4 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(3)
    y5 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(3)
    y6 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    y7 = torch.index_select(grid_normal_y.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    y8 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    intersection_grid_normal_y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 3) + (
            1 - mask.view(batchsize, height, width, 1).repeat(1, 1, 1, 8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    z2 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    z3 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    z4 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(3)
    z5 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                            x.shape).unsqueeze_(3)
    z6 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    z7 = torch.index_select(grid_normal_z.view(-1), 0,
                            z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    z8 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    intersection_grid_normal_z = torch.cat((z1, z2, z3, z4, z5, z6, z7, z8), 3) + (
            1 - mask.view(batchsize,height, width, 1).repeat(1, 1, 1, 8))

    # Change from grid coordinates to world coordinates
    voxel_min_point = Tensor(
        [bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos_bs(grid, intersection_pos_rough,
                                                voxel_min_point, voxel_min_point_index,
                                                ray_direction, voxel_size, mask,batchsize, width,height)

    intersection_pos = intersection_pos * mask.repeat(3, 1, 1, 1).permute(1, 2,3, 0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal_bs(intersection_grid_normal_x, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_y = get_intersection_normal_bs(intersection_grid_normal_y, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_z = get_intersection_normal_bs(intersection_grid_normal_z, intersection_pos, voxel_min_point,
                                                    voxel_size)

    # Put all the xyz-axis of the normal vectors into a single matrix
    intersection_normal_x_resize = intersection_normal_x.unsqueeze_(3)
    intersection_normal_y_resize = intersection_normal_y.unsqueeze_(3)
    intersection_normal_z_resize = intersection_normal_z.unsqueeze_(3)
    intersection_normal = torch.cat(
        (intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 3)
    intersection_normal = intersection_normal / torch.unsqueeze(torch.norm(intersection_normal, p=2, dim=3), 3).repeat(
        1, 1, 1, 3)

    return intersection_pos, intersection_normal


def refraction(l, normal, eta1, eta2):
    # l n x 3 x imHeight x imWidth
    # normal n x 3 x imHeight x imWidth
    # eta1 float
    # eta2 float
    cos_theta = torch.sum(l * (-normal), dim=2).unsqueeze(2)
    i_p = l + normal * cos_theta
    t_p = eta1 / eta2 * i_p

    t_p_norm = torch.sum(t_p * t_p, dim=2)
    totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(2)

    t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(2).expand_as(normal) * (-normal)
    t = t_i + t_p
    t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=2), min=1e-10)).unsqueeze(2)

    cos_theta_t = torch.sum(t * (-normal), dim=2).unsqueeze(2)

    e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
          torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
    e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
          torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

    attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()

    return t, attenuate, totalReflectMask

def refraction_bs(l, normal, eta1, eta2):
    # l [b,h,w,3]
    # normal [b,h,w,3]
    # eta1 float
    # eta2 float
    cos_theta = torch.sum(l * (-normal), dim=3).unsqueeze(3)
    i_p = l + normal * cos_theta
    t_p = eta1 / eta2 * i_p

    t_p_norm = torch.sum(t_p * t_p, dim=3)
    totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(3)

    t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(3).expand_as(normal) * (-normal)
    t = t_i + t_p
    t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=3), min=1e-10)).unsqueeze(3)

    cos_theta_t = torch.sum(t * (-normal), dim=3).unsqueeze(3)

    e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
          torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
    e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
          torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

    attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1)

    return t, attenuate, totalReflectMask


def reflection(l, normal):
    # l n x 3 x imHeight x imWidth
    # normal n x 3 x imHeight x imWidth
    # eta1 float
    # eta2 float
    cos_theta = torch.sum(l * (-normal), dim=2).unsqueeze(2)
    r_p = l + normal * cos_theta
    r_p_norm = torch.clamp(torch.sum(r_p * r_p, dim=2), 0, 0.999999)
    r_i = torch.sqrt(1 - r_p_norm).unsqueeze(2).expand_as(normal) * normal
    r = r_p + r_i
    r = r / torch.sqrt(torch.clamp(torch.sum(r * r, dim=2), min=1e-10).unsqueeze(2))

    return r

def reflection_bs(l, normal):
    # l n x 3 x imHeight x imWidth
    # normal n x 3 x imHeight x imWidth
    # eta1 float
    # eta2 float
    cos_theta = torch.sum(l * (-normal), dim=3).unsqueeze(3)
    r_p = l + normal * cos_theta
    r_p_norm = torch.clamp(torch.sum(r_p * r_p, dim=3), 0, 0.999999)
    r_i = torch.sqrt(1 - r_p_norm).unsqueeze(3).expand_as(normal) * normal
    r = r_p + r_i
    r = r / torch.sqrt(torch.clamp(torch.sum(r * r, dim=3), min=1e-10).unsqueeze(3))

    return r

def sampleEnvLight(l, envmap, envHeight, envWidth, imHeight, imWidth ):
    channelNum = envmap.size(2)

    l = torch.clamp(l, -0.999999, 0.999999)
    # Compute theta and phi
    x, y, z = torch.split(l, [1, 1, 1], dim=2 )
    theta = torch.acos(y )
    phi = torch.atan2( x, z )
    v = theta / np.pi * (envHeight-1)
    u = (-phi / np.pi / 2.0 + 0.5) * (envWidth-1)

    # Bilinear interpolation to get the new image


    u, v = torch.flatten(u), torch.flatten(v)
    u1 = torch.clamp(torch.floor(u).detach(), 0, envWidth-1)
    v1 = torch.clamp(torch.floor(v).detach(), 0, envHeight-1)
    u2 = torch.clamp(torch.ceil(u).detach(), 0, envWidth-1)
    v2 = torch.clamp(torch.ceil(v).detach(), 0, envHeight-1)

    w_r = (u - u1).unsqueeze(1)
    w_l = (1 - w_r )
    w_u = (v2 - v).unsqueeze(1)
    w_d = (1 - w_u )

    u1, v1 = u1.long(), v1.long()
    u2, v2 = u2.long(), v2.long()
    envmap = envmap.reshape([envHeight * envWidth, channelNum ])
    index = (v1 * envWidth + u2)
    envmap_ru = torch.index_select(envmap, 0, index )
    index = (v2 * envWidth + u2)
    envmap_rd = torch.index_select(envmap, 0, index )
    index = (v1 * envWidth + u1)
    envmap_lu = torch.index_select(envmap, 0, index )
    index = (v2 * envWidth + u1)
    envmap_ld = torch.index_select(envmap, 0, index )

    envmap_r = envmap_ru * w_u.expand_as(envmap_ru ) + \
            envmap_rd * w_d.expand_as(envmap_rd )
    envmap_l = envmap_lu * w_u.expand_as(envmap_lu ) + \
            envmap_ld * w_d.expand_as(envmap_ld )
    renderedImg = envmap_r * w_r.expand_as(envmap_r ) + \
            envmap_l * w_l.expand_as(envmap_l )

    # Post processing
    renderedImg = renderedImg.reshape([imHeight, imWidth, 3])

    return renderedImg

def sampleEnvLight_bs(l, envmap, envHeight, envWidth, batchsize, imHeight, imWidth ):
    offset = np.arange(0, batchsize).reshape([batchsize, 1, 1, 1])
    offset = (offset * envWidth * envHeight).astype(np.int64)
    offset = torch.from_numpy(offset)

    channelNum = envmap.size(3)

    l = torch.clamp(l, -0.999999, 0.999999)
    # Compute theta and phi
    x, y, z = torch.split(l, [1, 1, 1], dim=3)
    theta = torch.acos(y)
    phi = torch.atan2(x, z)
    watch = torch.sum(z==0)
    v = theta / np.pi * (envHeight - 1)
    u = (-phi / np.pi / 2.0 + 0.5) * (envWidth - 1)

    # Bilinear interpolation to get the new image
    offset = offset.detach()[0:batchsize, :]
    offset = offset.expand_as(u).clone().cuda()

    u, v = torch.flatten(u), torch.flatten(v)
    u1 = torch.clamp(torch.floor(u).detach(), 0, envWidth - 1)
    v1 = torch.clamp(torch.floor(v).detach(), 0, envHeight - 1)
    u2 = torch.clamp(torch.ceil(u).detach(), 0, envWidth - 1)
    v2 = torch.clamp(torch.ceil(v).detach(), 0, envHeight - 1)

    w_r = (u - u1).unsqueeze(1)
    w_l = (1 - w_r)
    w_u = (v2 - v).unsqueeze(1)
    w_d = (1 - w_u)

    u1, v1 = u1.long(), v1.long()
    u2, v2 = u2.long(), v2.long()
    offset = torch.flatten(offset)
    size_0 = envWidth * envHeight * batchsize
    envmap = envmap.reshape([size_0, channelNum])
    index = (v1 * envWidth + u2) + offset
    envmap_ru = torch.index_select(envmap, 0, index)
    index = (v2 * envWidth + u2) + offset
    envmap_rd = torch.index_select(envmap, 0, index)
    index = (v1 * envWidth + u1) + offset
    envmap_lu = torch.index_select(envmap, 0, index)
    index = (v2 * envWidth + u1) + offset
    envmap_ld = torch.index_select(envmap, 0, index)

    envmap_r = envmap_ru * w_u.expand_as(envmap_ru) + \
               envmap_rd * w_d.expand_as(envmap_rd)
    envmap_l = envmap_lu * w_u.expand_as(envmap_lu) + \
               envmap_ld * w_d.expand_as(envmap_ld)
    renderedImg = envmap_r * w_r.expand_as(envmap_r) + \
                  envmap_l * w_l.expand_as(envmap_l)

    # Post processing
    renderedImg = renderedImg.reshape([batchsize, imHeight, imWidth, channelNum])

    return renderedImg

def sampleEnvLight_near_bs(l, envmap, envHeight, envWidth, batchsize, imHeight, imWidth ):
    offset = np.arange(0, batchsize).reshape([batchsize, 1, 1, 1])
    offset = (offset * envWidth * envHeight).astype(np.int64)
    offset = torch.from_numpy(offset)

    channelNum = envmap.size(3)

    l = torch.clamp(l, -0.999999, 0.999999)
    # Compute theta and phi
    x, y, z = torch.split(l, [1, 1, 1], dim=3)
    theta = torch.acos(y)
    phi = torch.atan2(x, z)
    watch = torch.sum(z==0)
    v = theta / np.pi * (envHeight - 1)
    u = (-phi / np.pi / 2.0 + 0.5) * (envWidth - 1)

    # Bilinear interpolation to get the new image
    offset = offset.detach()[0:batchsize, :]
    offset = offset.expand_as(u).clone().cuda()

    u, v = torch.flatten(u), torch.flatten(v)
    u1 = torch.clamp(torch.floor(u).detach(), 0, envWidth - 1)
    v1 = torch.clamp(torch.floor(v).detach(), 0, envHeight - 1)
    u2 = torch.clamp(torch.ceil(u).detach(), 0, envWidth - 1)
    v2 = torch.clamp(torch.ceil(v).detach(), 0, envHeight - 1)

    w_r = torch.round((u - u1).unsqueeze(1))
    w_l = (1 - w_r)
    w_u = torch.round((v2 - v).unsqueeze(1))
    w_d = (1 - w_u)

    u1, v1 = u1.long(), v1.long()
    u2, v2 = u2.long(), v2.long()
    offset = torch.flatten(offset)
    size_0 = envWidth * envHeight * batchsize
    envmap = envmap.reshape([size_0, channelNum])
    index = (v1 * envWidth + u2) + offset
    envmap_ru = torch.index_select(envmap, 0, index)
    index = (v2 * envWidth + u2) + offset
    envmap_rd = torch.index_select(envmap, 0, index)
    index = (v1 * envWidth + u1) + offset
    envmap_lu = torch.index_select(envmap, 0, index)
    index = (v2 * envWidth + u1) + offset
    envmap_ld = torch.index_select(envmap, 0, index)

    envmap_r = envmap_ru * w_u.expand_as(envmap_ru) + \
               envmap_rd * w_d.expand_as(envmap_rd)
    envmap_l = envmap_lu * w_u.expand_as(envmap_lu) + \
               envmap_ld * w_d.expand_as(envmap_ld)
    renderedImg = envmap_r * w_r.expand_as(envmap_r) + \
                  envmap_l * w_l.expand_as(envmap_l)

    # Post processing
    renderedImg = renderedImg.reshape([batchsize, imHeight, imWidth, 3])

    return renderedImg

def transformCoordinate(batchSize, imHeight,imWidth, l, origin, lookat, up ):
    batchSize = origin.size(0 )
    assert(batchSize <= batchSize )

    # Rotate to world coordinate
    zAxis = origin - lookat
    yAxis = up
    xAxis = torch.cross(yAxis, zAxis, dim=1 )
    xAxis = xAxis / torch.sqrt(torch.clamp(torch.sum(xAxis * xAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )
    yAxis = yAxis / torch.sqrt(torch.clamp(torch.sum(yAxis * yAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )
    zAxis = zAxis / torch.sqrt(torch.clamp(torch.sum(zAxis * zAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )

    xAxis = xAxis.view([batchSize, 3, 1, 1, 1])
    yAxis = yAxis.view([batchSize, 3, 1, 1, 1])
    zAxis = zAxis.view([batchSize, 3, 1, 1, 1])
    rotMat = torch.cat([xAxis, yAxis, zAxis], dim=2 )
    l = l.unsqueeze(1)

    l = torch.sum(rotMat.expand([batchSize, 3, 3, imHeight, imWidth ] ) * \
            l.expand([batchSize, 3, 3, imHeight, imWidth ] ), dim=2)
    l = l / torch.sqrt( torch.clamp(torch.sum(l*l, dim=1 ).unsqueeze(1), min=1e-10 ) )

    return l

def getBackground_bs(batchSize,imHeight,imWidth,fov,envHeight,envWidth,origin, lookat, up, envmap):
    x, y = np.meshgrid(np.linspace(-1, 1, imWidth),
                       np.linspace(-1, 1, imHeight))
    fov = fov / 180 * np.pi
    tan_fovx = np.tan(fov/2)
    tan_fovy = tan_fovx / float(imWidth) * float(imHeight)
    x, y = x * tan_fovx, -y * tan_fovy
    z = -np.ones([imHeight, imWidth], dtype=np.float32)
    x, y, z = x[np.newaxis, :, :], y[np.newaxis, :, :], z[np.newaxis, :, :]
    x = -x
    v = np.concatenate([x, y, z], axis=0).astype(np.float32)
    v = v / np.maximum(np.sqrt(np.sum(v * v, axis=0))[np.newaxis, :], 1e-6)
    v = v[np.newaxis, :, :, :]
    v = torch.from_numpy(v).detach().cuda()

    # Compute the offset
    offset = np.arange(0, batchSize).reshape([batchSize, 1, 1, 1])
    offset = (offset * envWidth * envHeight).astype(np.int64)
    offset = torch.from_numpy(offset).detach().cuda()

    #l = v.expand_as(normal1).clone().cuda()
    l = v.repeat([batchSize,1,1,1])
    l = transformCoordinate(batchSize, imHeight,imWidth,l, origin, lookat, up).permute([0,2,3,1])
    backImg = sampleEnvLight_bs(l, envmap, envHeight, envWidth, batchSize, imHeight, imWidth)
    return backImg
