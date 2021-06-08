#-*- coding: utf-8 -*-
import os
import torch
from pytorch3d.io import load_obj, save_obj,load_ply,save_ply
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    point_mesh_distance
)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
os.environ['PYOPENGL_PLATFORM'] = 'egl'


mesh = trimesh.load('/mnt/data3/lj/transparent/Data/Shapes/train/Shape__0/visualHullSubd_20.ply')

points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000,surface_point_method='sample')

colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1


mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

#读取数据
gt_verts,gt_faces = load_ply('/mnt/data3/lj/transparent/Data/Shapes/train/Shape__0/poissonSubd.ply')
gt_face_idx = gt_faces.to(device)
gt_verts = gt_verts.to(device)
gt_meshes = Meshes([gt_verts],[gt_face_idx])

VH_verts,VH_faces = load_ply('/mnt/data3/lj/transparent/Data/Shapes/train/Shape__0/visualHullSubd_20.ply')
VH_faces_idx = VH_faces.to(device)
VH_verts = VH_verts.to(device)

#计算每个点到mesh距离
VH_num_max_points = VH_verts.size()[0]
VH_points_first = torch.tensor([0]).to(device)

gt_tris = gt_verts[gt_face_idx]
gt_tris_first_idx = VH_points_first

point2mesh_dis = point_mesh_distance.point_face_distance(VH_verts,VH_points_first,gt_tris,gt_tris_first_idx,VH_num_max_points)
#point2mesh_dis = point_mesh_distance.point_face_distance(gt_verts,VH_points_first,gt_tris,gt_tris_first_idx,gt_verts.size()[0])
array_point2mesh_dis = point2mesh_dis.to(torch.device('cpu'))
array_point2mesh_dis = array_point2mesh_dis.numpy()

point_mesh_distance.point_mesh_face_distance()
point_mesh_distance.point_face_distance()

