import numpy as np
from pano_calib.read_model import *
import cv2
import cupy as cp
import sys
from pano_calib.visualhull import *

def bilinear_interpolate(im, x, y):

    x0 = cp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = cp.floor(y).astype(int)
    y1 = y0 + 1

    x0 = cp.clip(x0, 0, im.shape[1]-1)
    x1 = cp.clip(x1, 0, im.shape[1]-1)
    y0 = cp.clip(y0, 0, im.shape[0]-1)
    y1 = cp.clip(y1, 0, im.shape[0]-1)

    Ia = im[(y0, x0)]
    Ib = im[ (y1, x0 )]
    Ic = im[ (y0, x1 )]
    Id = im[ (y1, x1 )]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa[:,:,cp.newaxis]*Ia + wb[:,:,cp.newaxis]*Ib + wc[:,:,cp.newaxis]*Ic + wd[:,:,cp.newaxis]*Id

def render_from_pano(R,intrinsic,width,height,pano_image,pano_width,pano_height):
    """
    render the calibrated image from the panorama image
    R(3,3),intrinsic(4) = [focal,cx,cy,k(distortion coefficient)]
    """
    #construct the coordinate grid
    coord_x = cp.linspace(0,width,width,dtype=cp.int)
    coord_y = cp.linspace(0,height,height,dtype=cp.int)
    coord_x,coord_y = cp.meshgrid(coord_x,coord_y,indexing='xy')

    if len(intrinsic)==3:
        coords = cp.stack([coord_x - intrinsic[1],coord_y - intrinsic[2],cp.ones_like(coord_x)*intrinsic[0]],axis=-1)[:,:,:,cp.newaxis]
    else:
        # coord_x = (coord_x - intrinsic[2])/intrinsic[0]
        # coord_y = (coord_y - intrinsic[3])/intrinsic[1]
        # r2 = coord_x*coord_x + coord_y * coord_y
        # coord_x = coord_x * (1 + intrinsic[3]*r2)
        # coord_y = coord_y * (1 + intrinsic[3] * r2)
        # coords = cp.stack([coord_x , coord_y , cp.ones_like(coord_x) ], axis=-1)[:, :, :, cp.newaxis]
        coords = cp.stack([(coord_x - intrinsic[2])/intrinsic[0], (coord_y - intrinsic[3])/intrinsic[1], cp.ones_like(coord_x)],
                          axis=-1)[:, :, :, cp.newaxis]

    #calculate the corresponding coordinate in the panorama image
    Rt_expand = cp.tile(R.transpose(),(height,width,1,1))
    coords_new = cp.matmul(Rt_expand,coords)
    coords_new = coords_new/cp.abs(cp.tile(coords_new[:,:,2:3,:],(1,1,3,1)))

    coord_new_x = coords_new[:,:,0,0]
    coord_new_y = coords_new[:, :, 1, 0]
    coord_new_z = coords_new[:, :, 2, 0]

    sinx = coord_new_x/cp.sqrt(coord_new_x*coord_new_x + coord_new_z*coord_new_z)
    siny = coord_new_y/cp.sqrt(coord_new_x*coord_new_x + coord_new_y*coord_new_y + coord_new_z*coord_new_z)

    theta = cp.arctan2(coord_new_x,coord_new_z)

    a = cp.where(theta<0)
    theta[cp.where(theta<0)] = 2 * cp.pi + theta[cp.where(theta<0)]

    phi = cp.arcsin(siny) + cp.pi/2

    X = theta/2/cp.pi * pano_width
    Y = phi/cp.pi * pano_height

    r_image = bilinear_interpolate(cp.asarray(pano_image),X,Y)

    return cp.asnumpy(r_image)


def bilinear_interpolate_np(im, x, y):

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[(y0, x0)]
    Ib = im[ (y1, x0 )]
    Ic = im[ (y0, x1 )]
    Id = im[ (y1, x1 )]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa[:,:,np.newaxis]*Ia + wb[:,:,np.newaxis]*Ib + wc[:,:,np.newaxis]*Ic + wd[:,:,np.newaxis]*Id

def render_from_pano_np(R,intrinsic,width,height,pano_image,pano_width,pano_height):
    """
    render the calibrated image from the panorama image
    R(3,3),intrinsic(4) = [focal,cx,cy,k(distortion coefficient)]
    """
    #construct the coordinate grid
    coord_x = np.linspace(0,width,width,dtype=cp.int)
    coord_y = np.linspace(0,height,height,dtype=cp.int)
    coord_x,coord_y = np.meshgrid(coord_x,coord_y,indexing='xy')

    if len(intrinsic)==3:
        coords = np.stack([coord_x - intrinsic[1],coord_y - intrinsic[2],np.ones_like(coord_x)*intrinsic[0]],axis=-1)[:,:,:,np.newaxis]
    else:
        # coord_x = (coord_x - intrinsic[2])/intrinsic[0]
        # coord_y = (coord_y - intrinsic[3])/intrinsic[1]
        # r2 = coord_x*coord_x + coord_y * coord_y
        # coord_x = coord_x * (1 + intrinsic[3]*r2)
        # coord_y = coord_y * (1 + intrinsic[3] * r2)
        # coords = cp.stack([coord_x , coord_y , cp.ones_like(coord_x) ], axis=-1)[:, :, :, cp.newaxis]
        coords = np.stack([(coord_x - intrinsic[2])/intrinsic[0], (coord_y - intrinsic[3])/intrinsic[1], np.ones_like(coord_x)],
                          axis=-1)[:, :, :, np.newaxis]

    #calculate the corresponding coordinate in the panorama image
    Rt_expand = np.tile(R.transpose(),(height,width,1,1))
    coords_new = np.matmul(Rt_expand,coords)
    coords_new = coords_new/np.abs(np.tile(coords_new[:,:,2:3,:],(1,1,3,1)))

    coord_new_x = coords_new[:,:,0,0]
    coord_new_y = coords_new[:, :, 1, 0]
    coord_new_z = coords_new[:, :, 2, 0]

    sinx = coord_new_x/np.sqrt(coord_new_x*coord_new_x + coord_new_z*coord_new_z)
    siny = coord_new_y/np.sqrt(coord_new_x*coord_new_x + coord_new_y*coord_new_y + coord_new_z*coord_new_z)

    theta = np.arctan2(coord_new_x,coord_new_z)

    a = np.where(theta<0)
    theta[np.where(theta<0)] = 2 * np.pi + theta[np.where(theta<0)]

    phi = np.arcsin(siny) + np.pi/2

    X = theta/2/np.pi * pano_width
    Y = phi/np.pi * pano_height

    r_image = bilinear_interpolate_np(pano_image,X,Y)

    return r_image


#------------------------------------------------------------------------------------------------------------------------
def main():

    # r0 = np.asarray([[1,0,0],
    #                  [0,1,0],
    #                  [0,0,1]])
    # r1 = np.asarray([[0, 0, -1],
    #                  [0, 1, 0],
    #                  [1, 0, 0]])
    # r2 = np.asarray([[-1, 0, 0],
    #                  [0, 1, 0],
    #                  [0, 0, -1]])
    # r3 = np.asarray([[0, 0, 1],
    #                  [0, 1, 0],
    #                  [-1, 0, 0]])
    #
    # pano_image = cv2.imread('G:\\transparent-real0\\court\\panorama.jpg')
    # image0 = render_from_pano_np(r0,[1901,2104,1560],4208,3120,pano_image,pano_image.shape[1],pano_image.shape[0])
    # image1 = render_from_pano_np(r1, [1901, 2104, 1560], 4208, 3120, pano_image, pano_image.shape[1],
    #                              pano_image.shape[0])
    # image2 = render_from_pano_np(r2, [1901, 2104, 1560], 4208, 3120, pano_image, pano_image.shape[1],
    #                              pano_image.shape[0])
    # image3 = render_from_pano_np(r3, [1901, 2104, 1560], 4208, 3120, pano_image, pano_image.shape[1],
    #                              pano_image.shape[0])
    # cv2.imwrite('G:\\transparent-real0\\court\\0.jpg', image0)
    # cv2.imwrite('G:\\transparent-real0\\court\\1.jpg', image1)
    # cv2.imwrite('G:\\transparent-real0\\court\\2.jpg', image2)
    # cv2.imwrite('G:\\transparent-real0\\court\\3.jpg', image3)
    #
    # return 0

    # if len(sys.argv) != 3:
    #     print("Usage: python read_model.py "
    #           "path/to/model/folder/txt"
    #           "the number of the anchor images")
    #     return

    # Rs = np.load(os.path.join(sys.argv[1], 'R.npy'))
    # ts = np.load(os.path.join(sys.argv[1], 't.npy'))
    # Rt = Rs.transpose((0,1,2))
    # cs = -np.matmul(Rs.transpose((0,1,2)),ts)
    # ranges = np.max(cs,axis=0) - np.min(cs,axis=0)

    path_to_model_txt_folder = 'G:\\transparent-real0\\court\\sparser'
    cameras, images, points3D = \
        read_model(path_to_model_txt_folder, ext=".bin")
    pano_image = cv2.imread('G:\\transparent-real0\\court\\panorama.jpg')
    mask_path = 'G:\\transparent-real0\\court\\masks'
    render_path = 'G:\\transparent-real0\\court\\watch'

    #use the anchor images to calculate the rotation and translation
    cs = []
    for i in range(1,int(sys.argv[2])+1):
        R = images[i].qvec2rotmat()
        t = np.array(images[i].tvec).reshape((3,1))
        c = np.matmul(-R.transpose(),t)
        cs.append(c)
    cs = np.concatenate(cs,axis=1)
    c_mean = np.mean(cs,axis=1,keepdims=True)
    RT_0 = images[1].qvec2rotmat().transpose()
    t0 = np.array(images[1].tvec).reshape((3,1))
    c0 = np.matmul(-RT_0,t0)

    Rs = []
    ts = []
    paras = []
    cs = []
    for i in range(1,len(images) + 1):
        R = images[i].qvec2rotmat()
        t = np.array(images[i].tvec).reshape((3, 1))
        c = np.matmul(-R.transpose(),t)

        c1 = np.matmul((c - c_mean).transpose(),RT_0).transpose()
        R1 = np.matmul(R,RT_0)
        Rs.append(R1)
        ts.append(np.matmul(R1,-c1))
        camera = cameras[images[i].camera_id]
        paras.append(np.array(camera.params))
        cs.append(c1)

        #render image
        #r_image = render_from_pano_np(R1,[1901,2104,1560],4208,3120,pano_image,pano_image.shape[1],pano_image.shape[0])
        # r_image = render_from_pano_np(R1, camera.params, camera.width, camera.height, pano_image, pano_image.shape[1],
        #                            pano_image.shape[0])
        # image_name = list(images[i].name)
        # image_name.insert(-4, '_r')
        # image_name = ''.join(image_name)
        #
        # cv2.imwrite(os.path.join(render_path,image_name),r_image)
    Rs = np.stack(Rs,axis=0)
    ts = np.stack(ts, axis=0)
    paras = np.stack(paras, axis=0)
    np.save(os.path.join(path_to_model_txt_folder,'R.npy'),Rs)
    np.save(os.path.join(path_to_model_txt_folder, 't.npy'),ts)
    np.save(os.path.join(path_to_model_txt_folder, 'intrinsic.npy'),paras)

    # start visual hull
    files = glob.glob(os.path.join(mask_path, '*.png'))
    files.sort()
    masks = []
    for file in files:
        masks.append(load_mask(file))

    voxel_binary = visualhull(3, 200, Rs[4:], ts[4:], np.tile(np.array([1901,2104,1560]),(len(masks),1)), masks, offset=[0, 0, 0])
    vertices, triangles = mcubes.marching_cubes(voxel_binary, 0.5)
    vertices = vertices * 3/(200-1) - 3/2
    write_voxels(cp.asnumpy(voxel_binary), os.path.join('G:\\transparent-real0\\court\\result', 'final.xyz'))
    write_mesh_ply(vertices, triangles, 'G:\\transparent-real0\\court\\result\\final.ply')





if __name__ == "__main__":
    main()