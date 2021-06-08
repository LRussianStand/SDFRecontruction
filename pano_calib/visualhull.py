import numpy as np
import cupy as cp
import cv2
import os
import glob
import mcubes
import trimesh
import plyfile
from pano_calib.read_model import *

def write_voxels(v,path):
    n = np.argwhere(v == 1)
    f = open(path, 'w')
    for p in n:
        f.write(f'{p[0]} {p[1]} {p[2]}')
        f.write('\n')
    f.close()

def write_mesh_ply(verts,faces,ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

def load_mask(file):

    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary,cv2.getStructuringElement(cv2.MORPH_RECT,(61,61)))
    # print('阈值：', ret)
    return binary/255

def grid_project(coords,R,t,intrinsic):
    """
    :param coords: (x1,x2,x3....,3) with 3 as (x,y,z)
    :param R: (3,3)
    :param t: (3,1)
    :param intrinsic:(f,cx,cy) or (fx,fy,cx,cy)
    :return:
    """
    if len(intrinsic) == 4:
        K = [[intrinsic[0],0,intrinsic[2]],
             [0,intrinsic[1],intrinsic[3]],
             [0,0,1]]
    else:
        K = [[intrinsic[0], 0, intrinsic[1]],
             [0, intrinsic[0], intrinsic[2]],
             [0, 0, 1]]

    K = cp.asarray(K,dtype=float)
    proj_mat = cp.matmul(K,cp.concatenate((R,t),axis=1))
    proj_mat = cp.tile(proj_mat,(coords.shape[0],coords.shape[1],coords.shape[2],1,1))

    proj_coord = cp.matmul(proj_mat,cp.concatenate((coords,cp.ones((coords.shape[0],coords.shape[1],coords.shape[2],1))),axis=-1)[:,:,:,:,cp.newaxis])
    proj_coord = proj_coord/(proj_coord[:,:,:,2:3] + 1e-10)

    return proj_coord[:,:,:,0:2,0]

def visualhull(length,gridsize,Rs,ts,paras,masks,offset = [0,0,0]):
    """
    N = number of the images
    :param Rs: (N,3,3)
    :param ts: (N,3,1)
    :param intrinsic: (fx,fy,cx,cy) or (f,cx,cy)
    :param mask: (N,height,width), the masks of the input images
    :return:
    """
    X = cp.linspace(0,length,gridsize) - length/2 - offset[0]
    Y = cp.linspace(0, length, gridsize) - length / 2 - offset[1]
    Z = cp.linspace(0, length, gridsize) - length / 2 - offset[2]
    coordx,coordy,coordz = cp.meshgrid(X,Y,Z,indexing='xy')
    coords = cp.stack((coordx,coordy,coordz),axis=-1)

    voxels = cp.ones((gridsize,gridsize,gridsize),dtype=float)
    for i in range(len(masks)):
        if i%4 != 0:
            continue
        R = cp.asarray(Rs[i,:,:])
        t = cp.asarray(ts[i,:,:])
        intrinsic = cp.asarray(paras[i,:])
        proj_coord = grid_project(coords,R,t,intrinsic)
        proj_x = cp.clip(proj_coord[:,:,:,0],0,masks[i].shape[1] - 1).astype(cp.int32)
        #proj_y = cp.clip(masks[i].shape[0] - 1 - proj_coord[:,:,:,1],0,masks[i].shape[0] - 1).astype(cp.int32)
        proj_y = cp.clip(proj_coord[:, :, :, 1], 0, masks[i].shape[0] - 1).astype(cp.int32)
        voxel_mask = cp.asarray(masks[i])[proj_y,proj_x]
        voxels = voxels * voxel_mask
        # write_voxels(cp.asnumpy(voxel_mask),os.path.join('G:\\transparent-real0\\corridor\\result',str(i)+'.xyz'))
        # write_voxels(cp.asnumpy(voxels), os.path.join('G:\\transparent-real0\\corridor\\result', str(i) + '_merge.xyz'))

    return cp.asnumpy(voxels)

def main():

    import sys
    # Rs = np.load(os.path.join(sys.argv[1], 'R.npy'))
    # ts = np.load(os.path.join(sys.argv[1], 't.npy'))
    # intrinsics = [1901,2104,1560]
    # files = glob.glob(os.path.join(sys.argv[2],'*.png'))
    # files.sort()


    model_path = 'G:\\transparent-real0\\corridor\\sparse'
    mask_path = 'G:\\transparent-real0\\corridor\\masks'
    cameras, images, points3D = \
        read_model(model_path, ext=".bin")

    Rs = []
    ts = []
    paras = []
    for i in range(1,len(images) + 1):
        R = images[i].qvec2rotmat()
        t = np.array(images[i].tvec).reshape((3, 1))
        Rs.append(R)
        ts.append(t)
        camera = cameras[images[i].camera_id]
        #paras.append(np.array(camera.params))
        paras.append(np.array([1901,2104,1560]))

    Rs = np.stack(Rs, axis=0)
    ts = np.stack(ts, axis=0)
    paras = np.stack(paras, axis=0)
    intrinsics = [1901,2104,1560]
    files = glob.glob(os.path.join(mask_path,'*.png'))
    files.sort()
    masks = []
    for file in files:
        masks.append(load_mask(file))

    # Rs = Rs[4:,:,:]
    # ts = ts[4:,:,:]
    voxel_binary = visualhull(3,200,Rs,ts,paras,masks,offset = [0,0,0])
    write_voxels(cp.asnumpy(voxel_binary), os.path.join('G:\\transparent-real0\\corridor\\result', 'final.xyz'))
    vertices, triangles = mcubes.marching_cubes(voxel_binary, 0.5)
    write_mesh_ply(vertices,triangles,'G:\\transparent-real0\\corridor\\result\\final.ply')
    return


if __name__ == "__main__":
    main()
    

