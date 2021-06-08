# -*- coding: utf-8 -*-
""" 
@Time    : 2021/4/28 9:48
@Author  : HCF
@FileName: testR.py
@SoftWare: PyCharm
"""
# -*- coding: utf-8 -*-
""" 
@Time    : 2021/4/8 21:24
@Author  : HCF
@FileName: test_rotation.py
@SoftWare: PyCharm
"""
import math
import cv2
import numpy as np
import glob
import os
import configPa

Parm = [3040, 6080]
image_size = [3120, 4208]
F = 1901
K = np.array([
        [1901.0, 0, 2104.0],
        [0, 1901.0, 1560.0],
        [0, 0, 1.0]])


fx = K[0][0]
fy = K[1][1]
cx = K[0][2]
cy = K[1][2]

def com(x, y, R, panorama):
    index = [x, y, F]
    a = R[0][0] * index[0] + R[0][1] * index[1] + R[0][2] * index[2]
    b = R[1][0] * index[0] + R[1][1] * index[1] + R[1][2] * index[2]
    c = R[2][0] * index[0] + R[2][1] * index[1] + R[2][2] * index[2]
    a = a / abs(c)
    b = b / abs(c)
    c = c / abs(c)
    sinX = a / math.sqrt(math.pow(a, 2) + math.pow(c, 2))
    sinY = b / math.sqrt(math.pow(a, 2) + math.pow(b, 2) + math.pow(c, 2))
    if a > 0 and c > 0:
        angle = math.asin(sinX)
    elif a < 0 and c > 0:
        angle =2 * math.pi + math.asin(sinX)
    elif a > 0 and c < 0:
        angle = math.pi - math.asin(sinX)
    elif a < 0 and c < 0:
        angle = math.pi - math.asin(sinX)
    else:
        angle = 0
    X_new = (angle / (2*math.pi)) * configPa.Parm[1]# 6080
    Y_new = ((math.asin(sinY) + math.pi/2)/math.pi) * configPa.Parm[0] #3040
    return panorama[int(Y_new)][int(X_new)]


def test_l_rotation(R, tag):
    image = cv2.imread('./image_whu\\IMG_20210201_165548.jpg')
    panorama = cv2.imread('./image_whu\\20210201_165006_553.jpg')
    for y in range(len(image)):
        for x in range(len(image[y])):
            temp = com((x - cx) * F / fx, (y - cy) * F / fy, R, panorama)
            image[y][x] = temp
    cv2.imwrite('1.jpg', image)

def load_data():
    files = glob.glob(os.path.join('./sparse', '*.npy'))
    print(files)
    with open(files[0], 'rb') as f:
         intrinsic = np.load(f)
    with open(files[1], 'rb') as f:
        R = np.load(f)
    print(intrinsic[0])
    # for i in range(5):
    #     test_l_rotation(np.transpose(R[22]), intrinsic[22][0], intrinsic[22][1],
    #                     intrinsic[22][2], str(i) + '-')
    test_l_rotation(np.transpose(R[21]), intrinsic[21][0], intrinsic[21][1],
                    intrinsic[21][2],  'transpose-21-')
    test_l_rotation(R[21], intrinsic[21][0], intrinsic[21][1],
                    intrinsic[21][2], '21-')
    test_l_rotation(R[21], 1901, intrinsic[21][1],
                    intrinsic[21][2], '1901-21-')
load_data()

#----------------------------------------------------------------------------------------------------------------------

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def render_from_pano(R,intrinsic,width,height,pano_image,pano_width,pano_height):
    """
    render the calibrated image from the panorama image
    R(3,3),intrinsic(4) = [focal,cx,cy,k(distortion coefficient)]
    """
    #construct the coordinate grid
    coord_x = np.linspace(0,width,dtype=np.int)
    coord_y = np.linspace(0,height,dtype=np.int)
    coord_x,coord_y = np.meshgrid(coord_x,coord_y)
    coords = np.stack([coord_x - intrinsic[1],coord_y - intrinsic[2],np.ones_like(coord_x)*intrinsic[0]],axis=-1)[:,:,:,np.newaxis]

    #calculate the corresponding coordinate in the panorama image
    R_expand = np.tile(R,(height,width,1,1))
    coords_new = np.matmul(R_expand,coords)
    coords_new = coords_new/np.abs(np.tile(coords_new[:,:,2:3,:],(1,1,3,1)))

    coord_new_x = coords_new[:,:,0,0]
    coord_new_y = coords_new[:, :, 1, 0]
    coord_new_z = coords_new[:, :, 2, 0]

    sinx = coord_new_x/np.sqrt(coord_new_x*coord_new_x + coord_new_z*coord_new_z)
    siny = coord_y/np.sqrt(coord_new_x*coord_new_x + coord_new_y*coord_new_y + coord_new_z*coord_new_z)

    theta = np.arctan2(coord_new_x,coord_new_z)
    theta[np.where(theta<0)] = 2 * np.pi + theta[np.where(theta<0)]
    phi = np.arcsin(siny) + np.pi/2

    X = theta/2/np.pi * pano_width
    Y = phi/np.pi * pano_height

    r_image = bilinear_interpolate(pano_image,X,Y)

    return r_image





