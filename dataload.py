# -*- coding: utf-8 -*-
""" 
@Time    : 2021/7/13 14:53
@Author  : HCF
@FileName: dataload.py
@SoftWare: PyCharm
"""
import numpy as np
import os.path as osp
from PIL import Image
import random
from paddle.io import Dataset
import os
import h5py
import cv2
import xml.etree.ElementTree as et
from mesh_to_sdf import mesh_to_sdf
import trimesh

#导入环境图 mesh to sdf 相机参数
class BatchLoaderMyreal(Dataset):
    def __init__(self, dataRoot, shapeRoot = None,
            imHeight = 360, imWidth = 480,
            envHeight = 256, envWidth = 512,
            isRandom=False, phase='TRAIN', rseed = 1,
            isLoadCam = False, isLoadEnvmap = False, tag=None, classNum=12,
            camNum = 10, shapeRs = 0, shapeRe = 1500, volumeSize=32, batchSize = None, isOptim = False, ignore = [],
                 isLoadSDF = True, grid_res = 8, bounding_radius = 1.1):

        self.dataRoot = dataRoot
        self.shapeRoot = shapeRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.phase = phase.upper()
        self.isLoadCam = isLoadCam
        self.isLoadEnvmap = isLoadEnvmap
        self.camNum = camNum
        self.shapeRs = shapeRs
        self.shapeRe = shapeRe
        self.isLoadSDF = isLoadSDF
        self.grid_res = grid_res
        self.bounding_radius = bounding_radius
        self.tag = tag
        self.classNum = classNum

        if batchSize is None:
            batchSize = camNum
            self.batchSize = min(batchSize , 10)
        else:
            self.batchSize = batchSize

        # self.minX, self.maxX = -1.1, 1.1
        # self.minY, self.maxY = -1.1, 1.1
        # self.minZ, self.maxZ = -1.1, 1.1
        # self.volumeSize = volumeSize
        # y, x, z = np.meshgrid(
        #     np.linspace(self.minX, self.maxX, volumeSize),
        #     np.linspace(self.minY, self.maxY, volumeSize),
        #     np.linspace(self.minZ, self.maxZ, volumeSize))
        # x = x[:, :, :, np.newaxis ]
        # y = y[:, :, :, np.newaxis ]
        # z = z[:, :, :, np.newaxis ]
        # coord = np.concatenate([x, y, z], axis=3 )

        # shapeList = sorted(glob.glob(osp.join(dataRoot) ))
        if isLoadCam:
            self.originArr = []
            self.lookatArr = []
            self.upArr = []
            for n in range(max(0, shapeRs ), min(classNum, shapeRe) ):
                if n in ignore:
                    continue
                shape = osp.join(shapeRoot, self.tag )
                if not osp.isdir(shape ):
                    continue
                camFileName = osp.join(shape, 'cam%d.txt' % camNum )
                with open(camFileName, 'r') as camIn:
                    camLines = camIn.readlines()
                viewNum = int(camLines[0].strip() )
                origins = []
                lookats = []
                ups = []
                for n in range(0, viewNum ):
                    originStr = camLines[3*n+1 ].strip().split(' ')
                    lookatStr = camLines[3*n+2 ].strip().split(' ')
                    upStr = camLines[3*n+3 ].strip().split(' ')

                    origin = np.array([float(x) for x in originStr ])[np.newaxis, :]
                    lookat = np.array([float(x) for x in lookatStr ])[np.newaxis, :]
                    up = np.array([float(x) for x in upStr])[np.newaxis, :]

                    origins.append(origin.astype(np.float32 ) )
                    lookats.append(lookat.astype(np.float32 ) )
                    ups.append(up.astype(np.float32 ) )

                origins = np.concatenate(origins, axis=0 )
                lookats = np.concatenate(lookats, axis=0 )
                ups = np.concatenate(ups, axis=0 )

                self.originArr.append(origins )
                self.lookatArr.append(lookats )
                self.upArr.append(ups )

        if isLoadEnvmap:
            self.envList = []
            self.scaleList = []
            envListUnique = []
            for n in range(max(0, shapeRs ), min(classNum, shapeRe ) ):
                if n in ignore:
                    continue
                shape = osp.join(shapeRoot, 'Shape__%d' % n )
                if not osp.isdir(shape ):
                    continue
                xmlFile = osp.join(shape, 'im.xml')
                # Create rendering file for Depth maps
                tree = et.parse(xmlFile )
                root = tree.getroot()

                shapes = root.findall('emitter')
                assert(len(shapes ) == 1 )
                for shape in shapes:
                    strings = shape.findall('string')
                    assert(len(strings) == 1 )
                    for st in strings:
                        envFileName = st.get('value')

                    envFileName = envFileName.replace('/home/zhl/CVPR20/TransparentShape','/mnt/data3/lzc/transparent')
                    if not osp.isfile(envFileName):
                        print(envFileName)
                    # if not envFileName.find('1640')==-1:
                    #     print(envFileName)
                    floats = shape.findall('float')
                    assert(len(floats) == 1 )
                    for f in floats:
                        scale = float(f.get('value') )
                    self.envList.append(envFileName )
                    self.scaleList.append(scale )

                    if envFileName not in envListUnique:
                        envListUnique.append(envFileName )
            print("Number of environment maps %d" % (len(envListUnique ) ) )



        if rseed is not None:
            random.seed(rseed)

        # Permute the image list
        self.count = camNum
        self.perm = list(range(self.count ) )
        if isRandom:
            random.shuffle(self.perm)




    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        # normalize the normal vector so that it will be unit length
        origins = []
        lookats = []
        ups = []

        envs = []
        shapeId = ind
        batchDict = {}
        batchDict['data_path'] = osp.join(self.dataRoot, self.tag)
        for imId in self.perm:
            if self.isLoadCam:
                origin = self.originArr[shapeId ][imId ]
                lookat = self.lookatArr[shapeId ][imId ]
                up = self.upArr[shapeId ][imId]

                origins.append(origin[np.newaxis, :] )
                lookats.append(lookat[np.newaxis, :] )
                ups.append(up[np.newaxis, :] )

            if self.isLoadEnvmap:
                envFileName = self.envList[shapeId ]
                scale = self.scaleList[shapeId ]
                env = cv2.imread(envFileName, -1)
                if env is None:
                    print(envFileName)
                env = env[:, :, ::-1]
                env = cv2.resize(env, (self.envWidth, self.envHeight ), interpolation=cv2.INTER_LINEAR)
                env = np.ascontiguousarray(env )
                env = env.transpose([2, 0, 1]) * scale

                envs.append(env[np.newaxis, :] )



        if self.isLoadCam:
            origins = np.concatenate(origins, axis=0 )
            lookats = np.concatenate(lookats, axis=0 )
            ups = np.concatenate(ups, axis=0 )

            batchDict['origin'] = origins
            batchDict['lookat'] = lookats
            batchDict['up'] = ups

        if self.isLoadEnvmap:
            envs = np.concatenate(envs, axis=0 )
            batchDict['env'] = envs

        #读取sdf文件
        if self.isLoadSDF:
            shapePath = osp.join(self.shapeRoot, "Shape__%d" % (shapeId + self.shapeRs))
            batchDict['shape_path'] = shapePath

            gt_sdfName = osp.join(shapePath, 'object_sdf_%d.npy'%(self.grid_res))
            if osp.isfile(gt_sdfName):
                batchDict['gt_grid'] = np.load(gt_sdfName).astype(np.float)
            else:
                gtName = osp.join(shapePath, 'meshGT_transform.ply')
                # gtName = osp.join(shapePath, 'object-1500000.obj')
                gtmesh = trimesh.load(gtName)
                linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, self.grid_res)
                grid_x, grid_y, grid_z = np.meshgrid(linear_space, linear_space, linear_space)
                coords = np.stack((grid_x, grid_y, grid_z), axis=3)
                query_points = coords.reshape((-1, 3))
                gtsdfs = mesh_to_sdf(gtmesh, query_points, surface_point_method='sample', sign_method='normal',
                                   bounding_radius=None, scan_count=100,
                                   scan_resolution=400, sample_point_count=10000000, normal_sample_count=20)
                gtsdfs = np.reshape(gtsdfs, grid_x.shape).transpose((1, 0, 2))
                batchDict['gt_grid'] = gtsdfs
                np.save(gt_sdfName, gtsdfs)

        return batchDict

    def loadHDR(self, imName, scale):
        if not osp.isfile(imName ):
            print('Error: %s does not exist.' % imName )
            assert(False )
        image = cv2.imread(imName, -1 )[:, :, ::-1]
        image = cv2.resize(image, (self.imWidth, self.imHeight ), interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image )
        imMean = np.mean(image )

        if scale is None:
            if self.phase == 'TRAIN':
                scale = (np.random.random() * 0.2 + 0.4) / imMean
            else:
                scale = 0.5 / imMean
        image = (image*scale).transpose([2, 0, 1] )
        return image, scale

    def loadImage(self, imName, isGama = False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            return im

        im = Image.open(imName)
        im = self.imResize(im)
        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        return im

    def imResize(self, im):
        w0, h0 = im.size
        if w0 != self.imHeight or h0 != self.imWidth:
            im = im.resize( (self.imWidth, self.imHeight ), Image.ANTIALIAS)
        return im