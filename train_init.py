# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import trimesh
import trimesh.repair
import argparse
from mesh_to_sdf import sample_sdf_near_surface, get_surface_point_cloud, mesh_to_sdf
import Network
from utils import *
from torch.utils.data import Dataset, DataLoader
import time
import torch.optim as optim
from mesh import create_mesh, convert_sdf_samples_to_ply
import numpy as np

from pytorch3d.io import load_obj, save_obj, load_ply, save_ply
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.transforms.transform3d import Transform3d,Scale,Translate
import utils
torch.set_printoptions(precision=8)


# 定义dataloader
class Points_dataset(Dataset):
    def __init__(self, points, sdfs):
        self.points = points
        self.sdfs = sdfs
        self.len = points.shape[0]

    def __getitem__(self, index):
        return self.points[index], self.sdfs[index]

    def __len__(self):
        return self.len


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', default='/mnt/data/lj/transparent/Data/Shapes/test/Shape__0/object.obj',
                    help='path to images')
parser.add_argument('--shapePath',
                    default='/mnt/data/lj/transparent/Data/Results/20-ssim_0.0001-12/128.pt',
                    help='path to images')
parser.add_argument('--batchsize', type=int, default=200000,
                    help='the number of sampled SDFs for each training iteration')
parser.add_argument('--epoch', type=int, default=10000, help='epoch number')
parser.add_argument('--ckpt_path', default='./ckpt', help='checkpoint path')

opt = parser.parse_args()

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]
paths = ['/mnt/data/lj/transparent/Data/Shapes/real/Shape__0/meshGT_transform.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__1/meshGT_transform.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__2/meshGT_transform.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__3/meshGT_transform.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__4/meshGT_transform.ply']
pathsvh = ['/mnt/data/lj/transparent/Data/Shapes/real/Shape__0/visualHullSubd_5.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__1/visualHullSubd_10.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__2/visualHullSubd_12.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__3/visualHullSubd_10.ply',
         '/mnt/data/lj/transparent/Data/Shapes/real/Shape__4/visualHullSubd_10.ply']

# for i in range(len(paths)):
#     path = paths[i]
#     path_vh = pathsvh[i]
#     verts_gt, faces_gt = load_ply(path)
#     faces_idx_gt = faces_gt.cuda().detach()
#     verts_gt = -verts_gt.cuda().detach()
#
#     verts_vh, faces_vh = load_ply(path_vh)
#     faces_idx_vh = faces_vh.cuda().detach()
#     verts_vh = -verts_vh.cuda().detach()
#
#     center = -verts_gt.mean(dim = 0)
#     t1 = Translate(center[0],center[1],center[2],device='cuda:0')
#     verts1 = t1.transform_points(verts_gt)
#     verts1_vh = t1.transform_points(verts_vh)
#
#     max_value = verts1.abs().max()
#     t2 = Scale(1/max_value,device='cuda:0')
#     verts2 = t2.transform_points(verts1)
#     verts2_vh = t2.transform_points(verts1_vh)
#
#     #faces_idx_gt = flip(faces_idx_gt,dim=1)
#     faces_idx_vh = flip(faces_idx_vh, dim=1)
#
#     save_obj(path_vh.replace('.ply','.obj'),verts2_vh,faces_idx_vh)
    #save_ply(path.replace('meshGT_transform.ply','object.ply'),verts2.cpu(),faces_gt.cpu(),ascii=True)

# paths = ['/mnt/data/lj/transparent/Data/Shapes/real/Shape__0/object.obj',
#          '/mnt/data/lj/transparent/Data/Shapes/real/Shape__1/object.obj',
#          '/mnt/data/lj/transparent/Data/Shapes/real/Shape__2/object.obj',
#          '/mnt/data/lj/transparent/Data/Shapes/real/Shape__3/object.obj',
#          '/mnt/data/lj/transparent/Data/Shapes/real/Shape__4/object.obj']
# for path in paths:
#     mesh = trimesh.load(path)
#     repaired = trimesh.repair.fill_holes(mesh)
#     mesh.export(path.replace('object.obj','object1.obj'))



def GenMeshfromPT(path, N):
    #grid_final = torch.load(path)
    #grid_final = torch.from_numpy(np.load(path)).float()
    #watch = grid_final.cpu().numpy()
    #watch_gt = np.load("/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_128.npy")
    grid_final = grid_construction_sphere_small(256, -1.1, 1.1, 1).cpu()
    voxel_origin = [-1.1, -1.1, -1.1]
    voxel_size = 2.2 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]
    samples = torch.cat((samples, grid_final.reshape(-1, 1).cpu()), dim=1)

    num_samples = N ** 3

    # samples.requires_grad = False

    convert_sdf_samples_to_ply(
        grid_final.data.cpu(),
        voxel_origin,
        voxel_size,
        path.replace('.pt', '.ply'),
    )

GenMeshfromPT('/mnt/data/lj/transparent/Data/Results/aaa/256.pt', 256)
#GenMeshfromPT('/mnt/data/lj/transparent/Data/Results/10-48-vh-0.0001-myreal-SGD-10imloss-ml1/128.pt', 128)

# GenMeshfromPT('/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_12.npy', 12)
# GenMeshfromPT('/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_24.npy', 24)
# GenMeshfromPT('/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_40.npy', 40)
# GenMeshfromPT('/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_48.npy', 48)
# GenMeshfromPT('/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_56.npy', 56)
# GenMeshfromPT('/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_64.npy', 64)
# GenMeshfromPT('/mnt/data/lj/transparent/Data/Shapes/myreal/Shape__0/object_sdf_128.npy', 128)


verts_gt, faces_gt, aux_gt = load_obj(opt.dataPath)
faces_idx_gt = faces_gt.verts_idx.to(device)
verts_gt = verts_gt.to(device)
verts_out, faces_out = load_ply(opt.shapePath.replace('.pt', '.ply'))
faces_idx_out = faces_out.to(device)
verts_out = verts_out.to(device)
verts_vh, faces_vh = load_ply(opt.dataPath.replace('object.obj', 'visualHullSubd_20.ply'))
faces_idx_vh = faces_vh.to(device)
verts_vh = verts_vh.to(device)

mesh_gt = Meshes(verts=[verts_gt], faces=[faces_idx_gt])
mesh_out = Meshes(verts=[verts_out], faces=[faces_idx_out])
mesh_vh = Meshes(verts=[verts_vh], faces=[faces_idx_vh])

sample_gt = sample_points_from_meshes(mesh_gt, int(1e6))
sample_out = sample_points_from_meshes(mesh_out, int(1e6))
sample_vh = sample_points_from_meshes(mesh_vh, int(1e6))

loss_chamfer_out, _ = chamfer_distance(sample_gt, sample_out)
loss_chamfer_vh, _ = chamfer_distance(sample_gt, sample_vh)
print("out:%.8f vs vh:%.8f" % (loss_chamfer_out.item(), loss_chamfer_vh.item()))

print(opt)
if not os.path.isdir(opt.ckpt_path):
    os.mkdir(opt.ckpt_path)

# 读取数据，获取采样点sdf
mesh = trimesh.load(opt.dataPath)
# pointcloud = get_surface_point_cloud(mesh, surface_point_method='sample', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=100000, calculate_normals=True)

# points, sdf = sample_sdf_near_surface(mesh, number_of_points=200000, surface_point_method='sample')  # (p,3),(p)


linear_space = torch.linspace(-1.1, 1.1, 64)
grid_x, grid_y, grid_z = torch.meshgrid(linear_space, linear_space, linear_space)
coords = torch.stack((grid_x, grid_y, grid_z), dim=3)
query_points = coords.view(-1, 3).numpy()
sdfs = mesh_to_sdf(mesh, query_points, surface_point_method='sample', sign_method='normal', bounding_radius=None,
                   scan_count=100,
                   scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
sdfs = np.reshape(sdfs, grid_x.size())

# 将采样点坐标 高纬转化
embed_xyz_fn, embed_xyz_dims = utils.get_embedder(10)

# 开始训练
dataloader = DataLoader(dataset=Points_dataset(points, sdf), batch_size=opt.batchsize, shuffle=True)

model = Network.TransSDF(embed_xyz_dims, 1, [256] * 8, [0, 1, 2, 3, 4, 5, 6, 7], 0.2, True, (0, 1, 2, 3, 4, 5, 6, 7),
                         [4])
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = torch.nn.DataParallel(model)
model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

start = time.time()
running_loss = 0.0
for epoch in range(opt.epoch):
    for i, data in enumerate(dataloader, 0):
        point, sdf = data

        input = point.cuda()
        gt = sdf.cuda()
        input = embed_xyz_fn(input)

        optimizer.zero_grad()

        outputs = model(input)
        loss = criterion(outputs, gt.view(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 50 == 49:
        # 每 100 次epoch
        print('[%d] loss: %.8f' % (epoch + 1, running_loss / 100))
        running_loss = 0.0
    if epoch % 300 == 299:
        # 保存模型
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'loss': loss.item(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(opt.ckpt_path, '%d' % (epoch) + '.pth.tar'))
        # 保存ply文件
        create_mesh(model, embed_xyz_fn, os.path.join(opt.ckpt_path, '%d' % (epoch) + '-2'))

print('Finished Training! Total cost time: ', time.time() - start)
