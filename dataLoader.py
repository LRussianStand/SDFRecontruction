#-*- coding: utf-8 -*-  可采用中文注释
import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
from paddle.io import Dataset
import os
import scipy.ndimage as ndimage
import h5py
import cv2
import xml.etree.ElementTree as et
from skimage.transform import resize
from mesh_to_sdf import mesh_to_sdf
import trimesh

#数据读取模块
class BatchLoader(Dataset):
    def __init__(self, dataRoot, shapeRoot = None,
            imHeight = 360, imWidth = 480,
            envHeight = 256, envWidth = 512,
            isRandom=False, phase='TRAIN', rseed = 1,
            isLoadVH = False, isLoadEnvmap = False,
            isLoadCam = False, isLoadOptim = False,
            camNum = 10, shapeRs = 0, shapeRe = 1500, volumeSize=32, batchSize = None, isOptim = False, ignore = [],
                 isLoadSDF = True, grid_res = 8, bounding_radius = 1.1):

        self.dataRoot = dataRoot
        self.shapeRoot = shapeRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.phase = phase.upper()
        self.isLoadVH = isLoadVH
        self.isLoadCam = isLoadCam
        self.isLoadEnvmap = isLoadEnvmap
        self.isLoadOptim = isLoadOptim
        self.camNum = camNum
        self.shapeRs = shapeRs
        self.shapeRe = shapeRe
        self.isOptim = isOptim
        self.isLoadSDF = isLoadSDF
        self.grid_res = grid_res
        self.bounding_radius = bounding_radius

        if batchSize is None:
            batchSize = camNum
            self.batchSize = min(batchSize , 10)
        else:
            self.batchSize = batchSize

        self.minX, self.maxX = -1.1, 1.1
        self.minY, self.maxY = -1.1, 1.1
        self.minZ, self.maxZ = -1.1, 1.1
        self.volumeSize = volumeSize
        y, x, z = np.meshgrid(
                np.linspace(self.minX, self.maxX, volumeSize ),
                np.linspace(self.minY, self.maxY, volumeSize ),
                np.linspace(self.minZ, self.maxZ, volumeSize ) )
        x = x[:, :, :, np.newaxis ]
        y = y[:, :, :, np.newaxis ]
        z = z[:, :, :, np.newaxis ]
        coord = np.concatenate([x, y, z], axis=3 )

        shapeList = glob.glob(osp.join(dataRoot, 'Shape__*') )
        if isLoadCam:
            self.originArr = []
            self.lookatArr = []
            self.upArr = []
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
                if n in ignore:
                    continue
                shape = osp.join(shapeRoot, 'Shape__%d' % n )
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
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
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

                    envFileName = envFileName.replace('/home/zhl/CVPR20/TransparentShape','../')
                    if not osp.isfile(envFileName):
                        print(shapeList[n])
                    if not envFileName.find('1640')==-1:
                        print(shapeList[n])
                    floats = shape.findall('float')
                    assert(len(floats) == 1 )
                    for f in floats:
                        scale = float(f.get('value') )
                    self.envList.append(envFileName )
                    self.scaleList.append(scale )

                    if envFileName not in envListUnique:
                        envListUnique.append(envFileName )
            print("Number of environment maps %d" % (len(envListUnique ) ) )

        self.imList = []
        for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
            if n in ignore:
                continue
            shape = osp.join(dataRoot, 'Shape__%d' % n )
            if not osp.isdir(shape ):
                continue
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.rgbe' ) ) )
            if isRandom:
                random.shuffle(imNames )
            if len(imNames ) < camNum:
                print('%s: %d' % (shape, len(imNames) ) )
                assert(False )
            self.imList.append(imNames[0:camNum ] )

        if rseed is not None:
            random.seed(rseed)

        # Permute the image list
        self.count = len(self.imList)
        self.perm = list(range(self.count ) )
        if isRandom:
            random.shuffle(self.perm)




    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        # normalize the normal vector so that it will be unit length

        imNames = self.imList[self.perm[ind ] ]
        if self.batchSize < self.camNum:
            random.shuffle(imNames )

            #当batchsize 为 camnum - 1时，最后一个照片替换为法向量camNum
            if self.isOptim:
                count = 0
                imNamesNew = []
                for n in range(0, self.camNum ):
                    isAdd = False
                    imName = imNames[n]
                    if n == self.camNum-2:
                        isAdd = True
                    else:
                        twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) ).replace('.rgbe', '.npy')
                        if not osp.isfile(twoNormalName ):
                            isAdd = True
                    if isAdd == True:
                        imNamesNew.append(imName )
                        count += 1
                    if count == self.batchSize:
                        break
                imNames = imNamesNew
            else:
                imNames = imNames[0:self.batchSize ]

        segs = []
        seg2s = []
        normals = []
        normal2s = []
        depths = []
        depth2s = []
        ims = []
        imEs = []

        origins = []
        lookats = []
        ups = []

        envs = []

        segVHs = []
        seg2VHs = []
        normalVHs = []
        normal2VHs = []
        depthVHs = []
        depth2VHs = []

        normalOpts = []
        normal2Opts = []

        imScale = None
        for imName in imNames:
            twoBounceName = imName.replace('im_', 'imtwoBounce_').replace('.rgbe', '.npy')
            if not osp.isfile(twoBounceName ):
                twoBounceName = imName.replace('im_', 'imtwoBounce_').replace('.rgbe', '.h5')
                hf = h5py.File(twoBounceName, 'r')
                twoBounce = np.array(hf.get('data'), dtype=np.float32 )
                hf.close()
            else:
                twoBounce = np.load(twoBounceName )

            if twoBounce.shape[0] != self.imWidth or twoBounce.shape[1] != self.imHeight:
                newTwoBounce1 = cv2.resize(twoBounce[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce2 = cv2.resize(twoBounce[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce3 = cv2.resize(twoBounce[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                newTwoBounce4 = cv2.resize(twoBounce[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce5 = cv2.resize(twoBounce[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce6 = cv2.resize(twoBounce[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                twoBounce = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                    newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

            normal = twoBounce[:, :, 0:3].transpose([2, 0, 1] )
            normal = np.ascontiguousarray(normal )
            seg = twoBounce[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
            seg = np.ascontiguousarray(seg.astype(np.float32) )
            seg = seg[:,:,::-1]

            depth = twoBounce[:, :, 3:6].transpose([2, 0, 1] )
            depth = np.ascontiguousarray(depth )
            depth = depth * seg

            normal2 = twoBounce[:, :, 7:10].transpose([2, 0, 1] )
            normal2 = np.ascontiguousarray(normal2 )
            seg2 = twoBounce[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
            seg2 = np.ascontiguousarray(seg2.astype(np.float32) )

            depth2 = twoBounce[:, :, 10:13].transpose([2, 0, 1] )
            depth2 = np.ascontiguousarray(depth2 )
            depth2 = depth2 * seg

            normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-10) )[np.newaxis, :]
            normal = normal * seg
            normal2 = normal2 / np.sqrt(np.maximum(np.sum(normal2 * normal2, axis=0), 1e-10) )[np.newaxis, :]
            normal2 = normal2 * seg


            # Read rendered images(照片value压缩到0~1)
            imE, imScale = self.loadHDR(imName, imScale )
            imE = imE[:,:,::-1]
            im = imE * seg

            imId = int(imName.split('/')[-1].split('.')[0].split('_')[-1]  )
            shapeId = int(imName.split('/')[-2].split('_')[-1] ) - self.shapeRs

            segs.append(seg[np.newaxis, :] )
            seg2s.append(seg2[np.newaxis, :] )
            normals.append(normal[np.newaxis, :] )
            normal2s.append(normal2[np.newaxis, :] )
            depths.append(depth[np.newaxis, :] )
            depth2s.append(depth2[np.newaxis, :] )
            ims.append(im[np.newaxis, :] )
            imEs.append(imE[np.newaxis, :] )

            # Load the rendering file
            if self.isLoadCam:
                origin = self.originArr[shapeId ][imId-1 ]
                lookat = self.lookatArr[shapeId ][imId-1 ]
                up = self.upArr[shapeId ][imId-1 ]

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
                env = env.transpose([2, 0, 1]) * imScale * scale

                envs.append(env[np.newaxis, :] )


            if self.isLoadVH:
                twoBounceVHName = imName.replace('im_', 'imVH_%dtwoBounce_' % self.camNum ).replace('.rgbe', '.npy')
                if not osp.isfile(twoBounceVHName ):
                    twoBounceVHName = imName.replace('im_', 'imVH_%dtwoBounce_' % self.camNum ).replace('.rgbe', '.h5')
                    hf = h5py.File(twoBounceVHName, 'r')
                    twoBounceVH = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoBounceVH = np.load(twoBounceVHName )

                if twoBounceVH.shape[0] != self.imWidth or twoBounceVH.shape[1] != self.imHeight:
                    newTwoBounce1 = cv2.resize(twoBounceVH[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce2 = cv2.resize(twoBounceVH[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce3 = cv2.resize(twoBounceVH[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    newTwoBounce4 = cv2.resize(twoBounceVH[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce5 = cv2.resize(twoBounceVH[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce6 = cv2.resize(twoBounceVH[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    twoBounceVH = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                        newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

                normalVH = twoBounceVH[:, :, 0:3].transpose([2, 0, 1])
                normalVH = np.ascontiguousarray(normalVH )
                segVH = twoBounceVH[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
                segVH = np.ascontiguousarray(segVH.astype(np.float32) )

                depthVH = twoBounceVH[:, :, 3:6].transpose([2, 0, 1])
                depthVH = np.ascontiguousarray(depthVH )
                depthVH = depthVH * segVH

                normal2VH = twoBounceVH[:, :, 7:10].transpose([2, 0, 1])
                normal2VH = np.ascontiguousarray(normal2VH )
                seg2VH = twoBounceVH[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
                seg2VH = np.ascontiguousarray(seg2VH.astype(np.float32) )

                depth2VH = twoBounceVH[:, :, 10:13].transpose([2, 0, 1])
                depth2VH = np.ascontiguousarray(depth2VH )
                depth2VH = depth2VH * segVH

                normalVH = normalVH / np.sqrt(np.maximum(np.sum(normalVH * normalVH, axis=0), 1e-10) )[np.newaxis, :]
                normalVH = normalVH * segVH
                normal2VH = normal2VH / np.sqrt(np.maximum(np.sum(normal2VH * normal2VH, axis=0), 1e-10) )[np.newaxis, :]
                normal2VH = normal2VH * segVH

                segVHs.append(segVH[np.newaxis, :] )
                seg2VHs.append(seg2VH[np.newaxis, :] )
                normalVHs.append(normalVH[np.newaxis, :] )
                normal2VHs.append(normal2VH[np.newaxis, :] )
                depthVHs.append(depthVH[np.newaxis, :] )
                depth2VHs.append(depth2VH[np.newaxis, :] )

            if self.isLoadOptim:
                twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) ).replace('.rgbe', '.npy')
                if not osp.isfile(twoNormalName ):
                    twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) ).replace('.rgbe', '.h5')
                    hf = h5py.File(twoNormalName, 'r')
                    twoNormals = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoNormals = np.load(twoNormalName )
                normalOpt, normal2Opt = twoNormals[:, :, 0:3], twoNormals[:, :, 3:6]
                normalOpt = cv2.resize(normalOpt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normal2Opt = cv2.resize(normal2Opt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normalOpt = np.ascontiguousarray(normalOpt.transpose([2, 0, 1] ) )
                normal2Opt = np.ascontiguousarray(normal2Opt.transpose([2, 0, 1] ) )

                normalOpt = normalOpt / np.sqrt(np.maximum(np.sum(normalOpt * normalOpt, axis=0), 1e-10) )[np.newaxis, :]
                normalOpt = normalOpt * seg
                normal2Opt = normal2Opt / np.sqrt(np.maximum(np.sum(normal2Opt * normal2Opt, axis=0), 1e-10) )[np.newaxis, :]
                normal2Opt = normal2Opt * seg

                normalOpts.append(normalOpt[np.newaxis, :] )
                normal2Opts.append(normal2Opt[np.newaxis, :] )

        segs = np.concatenate(segs, axis=0 )
        seg2s = np.concatenate(seg2s, axis=0 )
        normals = np.concatenate(normals, axis=0 )
        normal2s = np.concatenate(normal2s, axis=0 )
        depths = np.concatenate(depths, axis=0 )
        depth2s = np.concatenate(depth2s, axis=0 )
        ims = np.concatenate(ims, axis=0 )
        imEs = np.concatenate(imEs, axis=0 )

        batchDict = {'seg1': segs,
                'seg2': seg2s,
                'normal1': normals,
                'normal2': normal2s,
                'depth1': depths,
                'depth2': depth2s,
                'im':  ims,
                'imE': imEs,
                'name': imNames }

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

        if self.isLoadVH:
            segVHs = np.concatenate(segVHs, axis=0 )
            seg2VHs = np.concatenate(seg2VHs, axis=0 )
            normalVHs = np.concatenate(normalVHs, axis=0 )
            normal2VHs = np.concatenate(normal2VHs, axis=0 )
            depthVHs = np.concatenate(depthVHs, axis=0 )
            depth2VHs = np.concatenate(depth2VHs, axis=0 )

            batchDict['seg1VH'] = segVHs
            batchDict['seg2VH'] = seg2VHs
            batchDict['normal1VH'] = normalVHs
            batchDict['normal2VH'] = normal2VHs
            batchDict['depth1VH'] = depthVHs
            batchDict['depth2VH'] = depth2VHs

        if self.isLoadOptim:
            normalOpts = np.concatenate(normalOpts, axis=0 )
            normal2Opts = np.concatenate(normal2Opts, axis=0 )
            batchDict['normalOpt'] = normalOpts
            batchDict['normal2Opt'] = normal2Opts

        #读取sdf文件
        if self.isLoadSDF:
            imName = imNames[0]
            shapeId = imName.split('/')[-2]
            shapePath = osp.join(self.shapeRoot, shapeId)
            sdfName = osp.join(shapePath, 'visualHullSubd_%d_%d_sdf.npy' % (self.camNum,self.grid_res))
            batchDict['shape_path'] = shapePath
            if osp.isfile(sdfName):
                batchDict['grid'] = np.load(sdfName).astype(np.float)
            else:
                VHName = osp.join(shapePath, 'visualHullSubd_%d.ply' % self.camNum)
                mesh = trimesh.load(VHName)
                linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, self.grid_res)
                grid_x, grid_y, grid_z = np.meshgrid(linear_space, linear_space, linear_space)
                coords = np.stack((grid_x, grid_y, grid_z),axis=3)
                query_points = coords.reshape((-1,3))
                sdfs = mesh_to_sdf(mesh, query_points, surface_point_method='sample', sign_method='normal',
                                   bounding_radius=None, scan_count=100,
                                   scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
                sdfs = np.reshape(sdfs, grid_x.shape).transpose((1,0,2))
                batchDict['grid'] = sdfs
                np.save(sdfName,sdfs)

            gt_sdfName = osp.join(shapePath, 'object_sdf_%d.npy'%(self.grid_res))
            if osp.isfile(gt_sdfName):
                batchDict['gt_grid'] = np.load(gt_sdfName).astype(np.float)
            else:
                #gtName = osp.join(shapePath, 'meshGT_transform.ply')
                gtName = osp.join(shapePath, 'object.obj')
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
        #image = np.clip((image * scale), 0, 1).transpose([2, 0, 1])
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



#-------------------------------------------------------------------------------------------------------
class BatchLoaderReal2(Dataset):
    def __init__(self, dataRoot, shapeRoot = None,
            imHeight = 360, imWidth = 480,
            envHeight = 256, envWidth = 512,
            isRandom=False, phase='TRAIN', rseed = 1,
            isLoadVH = False, isLoadEnvmap = False,
            isLoadCam = False, isLoadOptim = False,
            camNum = 10, shapeRs = 0, shapeRe = 1500, volumeSize=32, batchSize = None, isOptim = False, ignore = [],
                 isLoadSDF = True, grid_res = 8, bounding_radius = 1.1):

        self.dataRoot = dataRoot
        self.shapeRoot = shapeRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.phase = phase.upper()
        self.isLoadVH = isLoadVH
        self.isLoadCam = isLoadCam
        self.isLoadEnvmap = isLoadEnvmap
        self.isLoadOptim = isLoadOptim
        self.camNum = camNum
        self.shapeRs = shapeRs
        self.shapeRe = shapeRe
        self.isOptim = isOptim
        self.isLoadSDF = isLoadSDF
        self.grid_res = grid_res
        self.bounding_radius = bounding_radius

        if batchSize is None:
            batchSize = camNum
            self.batchSize = min(batchSize , 10)
        else:
            self.batchSize = batchSize

        self.minX, self.maxX = -1.1, 1.1
        self.minY, self.maxY = -1.1, 1.1
        self.minZ, self.maxZ = -1.1, 1.1
        self.volumeSize = volumeSize
        y, x, z = np.meshgrid(
                np.linspace(self.minX, self.maxX, volumeSize ),
                np.linspace(self.minY, self.maxY, volumeSize ),
                np.linspace(self.minZ, self.maxZ, volumeSize ) )
        x = x[:, :, :, np.newaxis ]
        y = y[:, :, :, np.newaxis ]
        z = z[:, :, :, np.newaxis ]
        coord = np.concatenate([x, y, z], axis=3 )

        shapeList = glob.glob(osp.join(dataRoot, 'Shape__*') )
        if isLoadCam:
            self.originArr = []
            self.lookatArr = []
            self.upArr = []
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
                if n in ignore:
                    continue
                shape = osp.join(shapeRoot, 'Shape__%d' % n )
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
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
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

                    envFileName = envFileName.replace('/home/zhl/CVPR20/TransparentShape','../')
                    if not osp.isfile(envFileName):
                        print(shapeList[n])
                    if not envFileName.find('1640')==-1:
                        print(shapeList[n])
                    floats = shape.findall('float')
                    assert(len(floats) == 1 )
                    for f in floats:
                        scale = float(f.get('value') )
                    self.envList.append(envFileName )
                    self.scaleList.append(scale )

                    if envFileName not in envListUnique:
                        envListUnique.append(envFileName )
            print("Number of environment maps %d" % (len(envListUnique ) ) )

        self.imList = []
        for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
            if n in ignore:
                continue
            shape = osp.join(dataRoot, 'Shape__%d' % n )
            if not osp.isdir(shape ):
                continue
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.png' ) ) )
            if isRandom:
                random.shuffle(imNames )
            if len(imNames ) < camNum:
                print('%s: %d' % (shape, len(imNames) ) )
                assert(False )
            self.imList.append(imNames[0:camNum ] )

        if rseed is not None:
            random.seed(rseed)

        # Permute the image list
        self.count = len(self.imList)
        self.perm = list(range(self.count ) )
        if isRandom:
            random.shuffle(self.perm)




    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        # normalize the normal vector so that it will be unit length

        imNames = self.imList[self.perm[ind ] ]
        if self.batchSize < self.camNum:
            random.shuffle(imNames )

            #当batchsize 为 camnum - 1时，最后一个照片替换为法向量camNum
            if self.isOptim:
                count = 0
                imNamesNew = []
                for n in range(0, self.camNum ):
                    isAdd = False
                    imName = imNames[n]
                    if n == self.camNum-2:
                        isAdd = True
                    else:
                        twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) ).replace('.png', '.npy')
                        if not osp.isfile(twoNormalName ):
                            isAdd = True
                    if isAdd == True:
                        imNamesNew.append(imName )
                        count += 1
                    if count == self.batchSize:
                        break
                imNames = imNamesNew
            else:
                imNames = imNames[0:self.batchSize ]

        segs = []
        seg2s = []
        normals = []
        normal2s = []
        depths = []
        depth2s = []
        ims = []
        imEs = []

        origins = []
        lookats = []
        ups = []

        envs = []

        segVHs = []
        seg2VHs = []
        normalVHs = []
        normal2VHs = []
        depthVHs = []
        depth2VHs = []

        normalOpts = []
        normal2Opts = []

        imScale = None
        for imName in imNames:
            twoBounceName = imName.replace('im_', 'imVH_twoBounce_').replace('.png', '.npy')
            if not osp.isfile(twoBounceName ):
                twoBounceName = imName.replace('im_', 'imVH_twoBounce_').replace('.png', '.h5')
                hf = h5py.File(twoBounceName, 'r')
                twoBounce = np.array(hf.get('data'), dtype=np.float32 )
                hf.close()
            else:
                twoBounce = np.load(twoBounceName )

            if twoBounce.shape[0] != self.imWidth or twoBounce.shape[1] != self.imHeight:
                newTwoBounce1 = cv2.resize(twoBounce[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce2 = cv2.resize(twoBounce[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce3 = cv2.resize(twoBounce[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                newTwoBounce4 = cv2.resize(twoBounce[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce5 = cv2.resize(twoBounce[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce6 = cv2.resize(twoBounce[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                twoBounce = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                    newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

            normal = twoBounce[:, :, 0:3].transpose([2, 0, 1] )
            normal = np.ascontiguousarray(normal )
            seg = twoBounce[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
            seg = np.ascontiguousarray(seg.astype(np.float32) )
            seg = seg[:,:,::-1]

            depth = twoBounce[:, :, 3:6].transpose([2, 0, 1] )
            depth = np.ascontiguousarray(depth )
            depth = depth * seg

            normal2 = twoBounce[:, :, 7:10].transpose([2, 0, 1] )
            normal2 = np.ascontiguousarray(normal2 )
            seg2 = twoBounce[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
            seg2 = np.ascontiguousarray(seg2.astype(np.float32) )

            depth2 = twoBounce[:, :, 10:13].transpose([2, 0, 1] )
            depth2 = np.ascontiguousarray(depth2 )
            depth2 = depth2 * seg

            normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-10) )[np.newaxis, :]
            normal = normal * seg
            normal2 = normal2 / np.sqrt(np.maximum(np.sum(normal2 * normal2, axis=0), 1e-10) )[np.newaxis, :]
            normal2 = normal2 * seg


            # Read rendered images(照片value压缩到0~1)
            imE, imScale = self.loadHDR(imName, imScale )
            imE = imE[:,:,::-1]
            im = imE * seg

            imId = int(imName.split('/')[-1].split('.')[0].split('_')[-1]  )
            shapeId = int(imName.split('/')[-2].split('_')[-1] ) - self.shapeRs

            segs.append(seg[np.newaxis, :] )
            seg2s.append(seg2[np.newaxis, :] )
            normals.append(normal[np.newaxis, :] )
            normal2s.append(normal2[np.newaxis, :] )
            depths.append(depth[np.newaxis, :] )
            depth2s.append(depth2[np.newaxis, :] )
            ims.append(im[np.newaxis, :] )
            imEs.append(imE[np.newaxis, :] )

            # Load the rendering file
            if self.isLoadCam:
                origin = self.originArr[shapeId ][imId-1 ]
                lookat = self.lookatArr[shapeId ][imId-1 ]
                up = self.upArr[shapeId ][imId-1 ]

                origins.append(origin[np.newaxis, :] )
                lookats.append(lookat[np.newaxis, :] )
                ups.append(up[np.newaxis, :] )

            if self.isLoadEnvmap:
                envFileName = self.envList[shapeId ]
                scale = self.scaleList[shapeId ]
                env = cv2.imread(envFileName, -1)
                env = cv2.cvtColor(env,cv2.COLOR_BGRA2BGR)
                if env is None:
                    print(envFileName)
                env = env[:, :, ::-1]
                env = cv2.resize(env, (self.envWidth, self.envHeight ), interpolation=cv2.INTER_LINEAR)
                env = np.ascontiguousarray(env )
                env = env.transpose([2, 0, 1]) * imScale * scale

                envs.append(env[np.newaxis, :] )


            if self.isLoadVH:
                twoBounceVHName = imName.replace('im_', 'imVH_twoBounce_').replace('.png', '.npy')
                if not osp.isfile(twoBounceVHName ):
                    twoBounceVHName = imName.replace('im_', 'imVH_twoBounce_').replace('.png', '.h5')
                    hf = h5py.File(twoBounceVHName, 'r')
                    twoBounceVH = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoBounceVH = np.load(twoBounceVHName )

                if twoBounceVH.shape[0] != self.imWidth or twoBounceVH.shape[1] != self.imHeight:
                    newTwoBounce1 = cv2.resize(twoBounceVH[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce2 = cv2.resize(twoBounceVH[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce3 = cv2.resize(twoBounceVH[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    newTwoBounce4 = cv2.resize(twoBounceVH[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce5 = cv2.resize(twoBounceVH[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce6 = cv2.resize(twoBounceVH[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    twoBounceVH = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                        newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

                normalVH = twoBounceVH[:, :, 0:3].transpose([2, 0, 1])
                normalVH = np.ascontiguousarray(normalVH )
                segVH = twoBounceVH[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
                segVH = np.ascontiguousarray(segVH.astype(np.float32) )

                depthVH = twoBounceVH[:, :, 3:6].transpose([2, 0, 1])
                depthVH = np.ascontiguousarray(depthVH )
                depthVH = depthVH * segVH

                normal2VH = twoBounceVH[:, :, 7:10].transpose([2, 0, 1])
                normal2VH = np.ascontiguousarray(normal2VH )
                seg2VH = twoBounceVH[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
                seg2VH = np.ascontiguousarray(seg2VH.astype(np.float32) )

                depth2VH = twoBounceVH[:, :, 10:13].transpose([2, 0, 1])
                depth2VH = np.ascontiguousarray(depth2VH )
                depth2VH = depth2VH * segVH

                normalVH = normalVH / np.sqrt(np.maximum(np.sum(normalVH * normalVH, axis=0), 1e-10) )[np.newaxis, :]
                normalVH = normalVH * segVH
                normal2VH = normal2VH / np.sqrt(np.maximum(np.sum(normal2VH * normal2VH, axis=0), 1e-10) )[np.newaxis, :]
                normal2VH = normal2VH * segVH

                segVHs.append(segVH[np.newaxis, :] )
                seg2VHs.append(seg2VH[np.newaxis, :] )
                normalVHs.append(normalVH[np.newaxis, :] )
                normal2VHs.append(normal2VH[np.newaxis, :] )
                depthVHs.append(depthVH[np.newaxis, :] )
                depth2VHs.append(depth2VH[np.newaxis, :] )

            if self.isLoadOptim:
                twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) ).replace('.png', '.npy')
                if not osp.isfile(twoNormalName ):
                    twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) ).replace('.png', '.h5')
                    hf = h5py.File(twoNormalName, 'r')
                    twoNormals = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoNormals = np.load(twoNormalName )
                normalOpt, normal2Opt = twoNormals[:, :, 0:3], twoNormals[:, :, 3:6]
                normalOpt = cv2.resize(normalOpt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normal2Opt = cv2.resize(normal2Opt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normalOpt = np.ascontiguousarray(normalOpt.transpose([2, 0, 1] ) )
                normal2Opt = np.ascontiguousarray(normal2Opt.transpose([2, 0, 1] ) )

                normalOpt = normalOpt / np.sqrt(np.maximum(np.sum(normalOpt * normalOpt, axis=0), 1e-10) )[np.newaxis, :]
                normalOpt = normalOpt * seg
                normal2Opt = normal2Opt / np.sqrt(np.maximum(np.sum(normal2Opt * normal2Opt, axis=0), 1e-10) )[np.newaxis, :]
                normal2Opt = normal2Opt * seg

                normalOpts.append(normalOpt[np.newaxis, :] )
                normal2Opts.append(normal2Opt[np.newaxis, :] )

        segs = np.concatenate(segs, axis=0 )
        seg2s = np.concatenate(seg2s, axis=0 )
        normals = np.concatenate(normals, axis=0 )
        normal2s = np.concatenate(normal2s, axis=0 )
        depths = np.concatenate(depths, axis=0 )
        depth2s = np.concatenate(depth2s, axis=0 )
        ims = np.concatenate(ims, axis=0 )
        imEs = np.concatenate(imEs, axis=0 )

        batchDict = {'seg1': segs,
                'seg2': seg2s,
                'normal1': normals,
                'normal2': normal2s,
                'depth1': depths,
                'depth2': depth2s,
                'im':  ims,
                'imE': imEs,
                'name': imNames }

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

        if self.isLoadVH:
            segVHs = np.concatenate(segVHs, axis=0 )
            seg2VHs = np.concatenate(seg2VHs, axis=0 )
            normalVHs = np.concatenate(normalVHs, axis=0 )
            normal2VHs = np.concatenate(normal2VHs, axis=0 )
            depthVHs = np.concatenate(depthVHs, axis=0 )
            depth2VHs = np.concatenate(depth2VHs, axis=0 )

            batchDict['seg1VH'] = segVHs
            batchDict['seg2VH'] = seg2VHs
            batchDict['normal1VH'] = normalVHs
            batchDict['normal2VH'] = normal2VHs
            batchDict['depth1VH'] = depthVHs
            batchDict['depth2VH'] = depth2VHs

        if self.isLoadOptim:
            normalOpts = np.concatenate(normalOpts, axis=0 )
            normal2Opts = np.concatenate(normal2Opts, axis=0 )
            batchDict['normalOpt'] = normalOpts
            batchDict['normal2Opt'] = normal2Opts

        #读取sdf文件
        if self.isLoadSDF:
            imName = imNames[0]
            shapeId = imName.split('/')[-2]
            shapePath = osp.join(self.shapeRoot, shapeId)
            sdfName = osp.join(shapePath, 'visualHullSubd_%d_%d_sdf.npy' % (self.camNum,self.grid_res))
            batchDict['shape_path'] = shapePath
            if osp.isfile(sdfName):
                batchDict['grid'] = np.load(sdfName).astype(np.float)
            else:
                VHName = osp.join(shapePath, 'visualHullSubd_%d.ply' % self.camNum)
                mesh = trimesh.load(VHName)
                linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, self.grid_res)
                grid_x, grid_y, grid_z = np.meshgrid(linear_space, linear_space, linear_space)
                coords = np.stack((grid_x, grid_y, grid_z),axis=3)
                query_points = coords.reshape((-1,3))
                sdfs = mesh_to_sdf(mesh, query_points, surface_point_method='sample', sign_method='normal',
                                   bounding_radius=None, scan_count=100,
                                   scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
                sdfs = np.reshape(sdfs, grid_x.shape).transpose((1,0,2))
                batchDict['grid'] = sdfs
                np.save(sdfName,sdfs)

            gt_sdfName = osp.join(shapePath, 'object_sdf_%d.npy'%(self.grid_res))
            if osp.isfile(gt_sdfName):
                batchDict['gt_grid'] = np.load(gt_sdfName).astype(np.float)
            else:
                gtName = osp.join(shapePath, 'meshGT_transform.ply')
                gtmesh = trimesh.load(gtName)
                linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, self.grid_res)
                grid_x, grid_y, grid_z = np.meshgrid(linear_space, linear_space, linear_space)
                coords = np.stack((grid_x, grid_y, grid_z), axis=3)
                query_points = coords.reshape((-1, 3))
                gtsdfs = mesh_to_sdf(gtmesh, query_points, surface_point_method='sample', sign_method='normal',
                                   bounding_radius=None, scan_count=100,
                                   scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
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
        #image = np.clip((image * scale), 0, 1).transpose([2, 0, 1])
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

#------------------------------------------------------------------------------------
class BatchLoaderReal(Dataset):
    def __init__(self, dataRoot, shapeRoot = None,
            imHeight = 360, imWidth = 480,
            envHeight = 256, envWidth = 512,
            isRandom=True, phase='TRAIN', rseed = 1,
            isLoadVH = False, isLoadEnvmap = False,
            isLoadCam = False, isLoadOptim = False,
            camNum = 10, shapeRs = 0, shapeRe = 1500, volumeSize=32, batchSize = None, isOptim = False,
            ignore = [],isLoadSDF = True, grid_res = 8, bounding_radius = 1.1,):

        self.dataRoot = dataRoot
        self.shapeRoot = shapeRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.phase = phase.upper()
        self.isLoadVH = isLoadVH
        self.isLoadCam = isLoadCam
        self.isLoadEnvmap = isLoadEnvmap
        self.isLoadOptim = isLoadOptim
        #self.camNum = camNum
        self.shapeRs = shapeRs
        self.shapeRe = shapeRe
        self.isOptim = isOptim
        self.isLoadSDF = isLoadSDF
        self.grid_res = grid_res
        self.bounding_radius = bounding_radius

        if batchSize is None:
            batchSize = camNum
            self.batchSize = min(batchSize , 10)
        else:
            self.batchSize = batchSize

        self.minX, self.maxX = -1.1, 1.1
        self.minY, self.maxY = -1.1, 1.1
        self.minZ, self.maxZ = -1.1, 1.1
        self.volumeSize = volumeSize
        y, x, z = np.meshgrid(
                np.linspace(self.minX, self.maxX, volumeSize ),
                np.linspace(self.minY, self.maxY, volumeSize ),
                np.linspace(self.minZ, self.maxZ, volumeSize ) )
        x = x[:, :, :, np.newaxis ]
        y = y[:, :, :, np.newaxis ]
        z = z[:, :, :, np.newaxis ]
        coord = np.concatenate([x, y, z], axis=3 )

        shapeList = sorted(glob.glob(osp.join(dataRoot, 'Shape__*') ))
        self.camNumList = []
        if isLoadCam:
            self.originArr = []
            self.lookatArr = []
            self.upArr = []
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
                if n in ignore:
                    continue
                shape = osp.join(shapeRoot, 'Shape__%d' % n )
                if not osp.isdir(shape ):
                    continue
                print(shape)
                camNum = int(glob.glob(osp.join(shape, 'visualHullSubd*.ply'))[0].split('_')[-1].split('.')[0])
                self.camNumList.append(camNum)
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
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
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
                    envFileName = envFileName.replace('/home/zhl/CVPR20/TransparentShape', '../')

                    floats = shape.findall('float')
                    assert(len(floats) == 1 )
                    for f in floats:
                        scale = float(f.get('value') )
                    self.envList.append(envFileName )
                    self.scaleList.append(scale )

                    if envFileName not in envListUnique:
                        envListUnique.append(envFileName )
            print("Number of environment maps %d" % (len(envListUnique ) ) )

        self.imList = []
        self.shapeNameList = []
        self.camNumSingleList = []
        for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
            if n in ignore:
                continue
            shape = osp.join(dataRoot, 'Shape__%d' % n )
            if not osp.isdir(shape ):
                continue
            camNum = self.camNumList[n]
            #imNames = sorted(glob.glob(osp.join(shape, 'im_*.rgbe' ) ) )
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.png' ) ) )
            #random.shuffle(imNames )
            if self.batchSize > 1:
                if len(imNames ) < camNum:
                    print('%s: %d' % (shape, len(imNames) ) )
                    assert(False )
                self.imList.append(imNames[0:camNum ] )
                self.shapeNameList.append(os.path.basename(shape))
            elif self.batchSize == 1:
                for imName in imNames:
                    self.imList.append([imName])
                    self.camNumSingleList.append(camNum)
                    self.shapeNameList.append(os.path.basename(shape))

        if rseed is not None:
            random.seed(rseed)

        # Permute the image list
        self.count = len(self.imList)
        self.perm = list(range(self.count ) )
        if isRandom:
            random.shuffle(self.perm)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        # normalize the normal vector so that it will be unit length

        #if self.batchSize < self.camNum:
        if self.batchSize > 1:
            camNum = self.camNumList[self.perm[ind ]]
        elif self.batchSize == 1:
            camNum = self.camNumSingleList[self.perm[ind ]]

        shapeName = self.shapeNameList[self.perm[ind ]]

        imNames = self.imList[self.perm[ind ] ]
        if self.batchSize < camNum:
            #random.shuffle(imNames )
            if self.isOptim:
                count = 0
                imNamesNew = []
                #for n in range(0, self.camNum ):
                for n in range(0, camNum ):
                    isAdd = False
                    imName = imNames[n]
                    if camNum - n == self.batchSize - count:
                        isAdd = True
                    else:
                        twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (camNum ) ).replace('.rgbe', '.npy')
                        if not osp.isfile(twoNormalName ):
                            isAdd = True
                    if isAdd == True:
                        imNamesNew.append(imName )
                        count += 1
                    if count == self.batchSize:
                        break
                imNames = imNamesNew
            else:
                imNames = imNames[0:self.batchSize ]

        segs = []
        #seg2s = []
        #normals = []
        #normal2s = []
        depths = []
        depth2s = []
        ims = []
        imEs = []

        origins = []
        lookats = []
        ups = []

        envs = []

        segVHs = []
        seg2VHs = []
        normalVHs = []
        normal2VHs = []
        depthVHs = []
        depth2VHs = []

        normalOpts = []
        normal2Opts = []

        imScale = None
        for imName in imNames:
            '''
            twoBounceName = imName.replace('im_', 'imtwoBounce_').replace('.rgbe', '.npy')
            if not osp.isfile(twoBounceName ):
                twoBounceName = imName.replace('im_', 'imtwoBounce_').replace('.rgbe', '.h5')
                hf = h5py.File(twoBounceName, 'r')
                twoBounce = np.array(hf.get('data'), dtype=np.float32 )
                hf.close()
            else:
                twoBounce = np.load(twoBounceName )

            if twoBounce.shape[0] != self.imWidth or twoBounce.shape[1] != self.imHeight:
                newTwoBounce1 = cv2.resize(twoBounce[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce2 = cv2.resize(twoBounce[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce3 = cv2.resize(twoBounce[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                newTwoBounce4 = cv2.resize(twoBounce[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce5 = cv2.resize(twoBounce[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce6 = cv2.resize(twoBounce[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                twoBounce = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                    newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

            normal = twoBounce[:, :, 0:3].transpose([2, 0, 1] )
            normal = np.ascontiguousarray(normal )
            seg = twoBounce[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
            seg = np.ascontiguousarray(seg.astype(np.float32) )

            depth = twoBounce[:, :, 3:6].transpose([2, 0, 1] )
            depth = np.ascontiguousarray(depth )
            depth = depth * seg

            normal2 = twoBounce[:, :, 7:10].transpose([2, 0, 1] )
            normal2 = np.ascontiguousarray(normal2 )
            seg2 = twoBounce[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
            seg2 = np.ascontiguousarray(seg2.astype(np.float32) )

            depth2 = twoBounce[:, :, 10:13].transpose([2, 0, 1] )
            depth2 = np.ascontiguousarray(depth2 )
            depth2 = depth2 * seg

            normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-10) )[np.newaxis, :]
            normal = normal * seg
            normal2 = normal2 / np.sqrt(np.maximum(np.sum(normal2 * normal2, axis=0), 1e-10) )[np.newaxis, :]
            normal2 = normal2 * seg
            '''

            # Read rendered images
            #imE, imScale = self.loadHDR(imName, imScale )
            imE = self.loadImage(imName, False )
            imE = imE[:,:,::-1]
            seg = self.loadMask(imName.replace('im', 'seg')).astype(np.float32) / 255
            seg = seg[:,:,::-1]
            im = imE * seg

            imId = int(imName.split('/')[-1].split('.')[0].split('_')[-1]  )
            shapeId = int(imName.split('/')[-2].split('_')[-1] ) - self.shapeRs

            segs.append(seg[np.newaxis, :] )
            #seg2s.append(seg2[np.newaxis, :] )
            #normals.append(normal[np.newaxis, :] )
            #normal2s.append(normal2[np.newaxis, :] )
            #depths.append(depth[np.newaxis, :] )
            #depth2s.append(depth2[np.newaxis, :] )
            ims.append(im[np.newaxis, :] )
            imEs.append(imE[np.newaxis, :] )

            # Load the rendering file
            if self.isLoadCam:
                origin = self.originArr[shapeId ][imId-1 ]
                lookat = self.lookatArr[shapeId ][imId-1 ]
                up = self.upArr[shapeId ][imId-1 ]
                origins.append(origin[np.newaxis, :] )
                lookats.append(lookat[np.newaxis, :] )
                ups.append(up[np.newaxis, :] )

            if self.isLoadEnvmap:
                envFileName = self.envList[shapeId ]
                #scale = self.scaleList[shapeId ]
                env = cv2.cvtColor(cv2.imread(envFileName, -1), cv2.COLOR_BGRA2BGR)[:, :, ::-1]
                #env = cv2.imread(envFileName, -1)[:, :, ::-1]
                env = cv2.resize(env, (self.envWidth, self.envHeight ), interpolation=cv2.INTER_LINEAR)
                env = np.ascontiguousarray(env )
                env = env / 255
                #env = env.transpose([2, 0, 1]) * imScale * scale
                env = env.transpose([2, 0, 1]).astype(np.float32)

                envs.append(env[np.newaxis, :] )


            if self.isLoadVH:
                #twoBounceVHName = imName.replace('im_', 'imVH_%dtwoBounce_' % self.camNum ).replace('.png', '.npy')
                twoBounceVHName = imName.replace('im_', 'imVH_twoBounce_' ).replace('.png', '.npy')
                if not osp.isfile(twoBounceVHName ):
                    twoBounceVHName = imName.replace('im_', 'imVH_twoBounce_' ).replace('.png', '.h5')
                    hf = h5py.File(twoBounceVHName, 'r')
                    twoBounceVH = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoBounceVH = np.load(twoBounceVHName )

                if twoBounceVH.shape[0] != self.imWidth or twoBounceVH.shape[1] != self.imHeight:
                    newTwoBounce1 = cv2.resize(twoBounceVH[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce2 = cv2.resize(twoBounceVH[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce3 = cv2.resize(twoBounceVH[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    newTwoBounce4 = cv2.resize(twoBounceVH[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce5 = cv2.resize(twoBounceVH[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce6 = cv2.resize(twoBounceVH[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    twoBounceVH = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                        newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

                normalVH = twoBounceVH[:, :, 0:3].transpose([2, 0, 1])
                normalVH = np.ascontiguousarray(normalVH )
                segVH = twoBounceVH[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
                segVH = np.ascontiguousarray(segVH.astype(np.float32) )

                depthVH = twoBounceVH[:, :, 3:6].transpose([2, 0, 1])
                depthVH = np.ascontiguousarray(depthVH )
                depthVH = depthVH * segVH

                normal2VH = twoBounceVH[:, :, 7:10].transpose([2, 0, 1])
                normal2VH = np.ascontiguousarray(normal2VH )
                seg2VH = twoBounceVH[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
                seg2VH = np.ascontiguousarray(seg2VH.astype(np.float32) )

                depth2VH = twoBounceVH[:, :, 10:13].transpose([2, 0, 1])
                depth2VH = np.ascontiguousarray(depth2VH )
                depth2VH = depth2VH * segVH

                normalVH = normalVH / np.sqrt(np.maximum(np.sum(normalVH * normalVH, axis=0), 1e-10) )[np.newaxis, :]
                normalVH = normalVH * segVH
                normal2VH = normal2VH / np.sqrt(np.maximum(np.sum(normal2VH * normal2VH, axis=0), 1e-10) )[np.newaxis, :]
                normal2VH = normal2VH * segVH

                segVHs.append(segVH[np.newaxis, :] )
                seg2VHs.append(seg2VH[np.newaxis, :] )
                normalVHs.append(normalVH[np.newaxis, :] )
                normal2VHs.append(normal2VH[np.newaxis, :] )
                depthVHs.append(depthVH[np.newaxis, :] )
                depth2VHs.append(depth2VH[np.newaxis, :] )

            if self.isLoadOptim:
                twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (camNum ) ).replace('.rgbe', '.npy')
                if not osp.isfile(twoNormalName ):
                    twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (camNum ) ).replace('.rgbe', '.h5')
                    hf = h5py.File(twoNormalName, 'r')
                    twoNormals = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoNormals = np.load(twoNormalName )
                normalOpt, normal2Opt = twoNormals[:, :, 0:3], twoNormals[:, :, 3:6]
                normalOpt = cv2.resize(normalOpt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normal2Opt = cv2.resize(normal2Opt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normalOpt = np.ascontiguousarray(normalOpt.transpose([2, 0, 1] ) )
                normal2Opt = np.ascontiguousarray(normal2Opt.transpose([2, 0, 1] ) )

                normalOpt = normalOpt / np.sqrt(np.maximum(np.sum(normalOpt * normalOpt, axis=0), 1e-10) )[np.newaxis, :]
                normalOpt = normalOpt * seg
                normal2Opt = normal2Opt / np.sqrt(np.maximum(np.sum(normal2Opt * normal2Opt, axis=0), 1e-10) )[np.newaxis, :]
                normal2Opt = normal2Opt * seg

                normalOpts.append(normalOpt[np.newaxis, :] )
                normal2Opts.append(normal2Opt[np.newaxis, :] )

        segs = np.concatenate(segs, axis=0 )
        #seg2s = np.concatenate(seg2s, axis=0 )
        #normals = np.concatenate(normals, axis=0 )
        #normal2s = np.concatenate(normal2s, axis=0 )
        #depths = np.concatenate(depths, axis=0 )
        #depth2s = np.concatenate(depth2s, axis=0 )
        ims = np.concatenate(ims, axis=0 )
        imEs = np.concatenate(imEs, axis=0 )

        batchDict = {
                'seg1': segs,
                #'seg2': seg2s,
                #'normal1': normals,
                #'normal2': normal2s,
                #'depth1': depths,
                #'depth2': depth2s,
                'im':  ims,
                'imE': imEs,
                'name': imNames,
                'camNum': camNum,
                'shapeName': shapeName}

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

        if self.isLoadVH:
            segVHs = np.concatenate(segVHs, axis=0 )
            seg2VHs = np.concatenate(seg2VHs, axis=0 )
            normalVHs = np.concatenate(normalVHs, axis=0 )
            normal2VHs = np.concatenate(normal2VHs, axis=0 )
            depthVHs = np.concatenate(depthVHs, axis=0 )
            depth2VHs = np.concatenate(depth2VHs, axis=0 )

            batchDict['seg1VH'] = segVHs
            batchDict['seg2VH'] = seg2VHs
            batchDict['normal1VH'] = normalVHs
            batchDict['normal2VH'] = normal2VHs
            batchDict['depth1VH'] = depthVHs
            batchDict['depth2VH'] = depth2VHs

        if self.isLoadOptim:
            normalOpts = np.concatenate(normalOpts, axis=0 )
            normal2Opts = np.concatenate(normal2Opts, axis=0 )
            batchDict['normalOpt'] = normalOpts
            batchDict['normal2Opt'] = normal2Opts

        # 读取sdf文件
        if self.isLoadSDF:
            imName = imNames[0]
            shapeId = imName.split('/')[-2]
            shapePath = osp.join(self.shapeRoot, shapeId)
            sdfName = osp.join(shapePath, 'visualHullSubd_%d_%d_sdf.npy' % (camNum, self.grid_res))
            batchDict['shape_path'] = shapePath
            if osp.isfile(sdfName):
                batchDict['grid'] = np.load(sdfName).astype(np.float)
            else:
                VHName = osp.join(shapePath, 'visualHullSubd_%d.ply' % camNum)
                mesh = trimesh.load(VHName)
                linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, self.grid_res)
                grid_x, grid_y, grid_z = np.meshgrid(linear_space, linear_space, linear_space)
                coords = np.stack((grid_x, grid_y, grid_z), axis=3)
                query_points = coords.reshape((-1, 3))
                sdfs = mesh_to_sdf(mesh, query_points, surface_point_method='sample', sign_method='normal',
                                   bounding_radius=None, scan_count=100,
                                   scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
                sdfs = np.reshape(sdfs, grid_x.shape).transpose((1, 0, 2))
                batchDict['grid'] = sdfs
                np.save(sdfName, sdfs)

            gt_sdfName = osp.join(shapePath, 'object_sdf_%d.npy' % (self.grid_res))
            if osp.isfile(gt_sdfName):
                batchDict['gt_grid'] = np.load(gt_sdfName).astype(np.float)
            else:
                gtName = osp.join(shapePath, 'meshGT_transform.ply')
                gtmesh = trimesh.load(gtName)
                linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, self.grid_res)
                grid_x, grid_y, grid_z = np.meshgrid(linear_space, linear_space, linear_space)
                coords = np.stack((grid_x, grid_y, grid_z), axis=3)
                query_points = coords.reshape((-1, 3))
                gtsdfs = mesh_to_sdf(gtmesh, query_points, surface_point_method='sample', sign_method='normal',
                                     bounding_radius=None, scan_count=100,
                                     scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
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
        image = np.clip( (image*scale), 0, 1).transpose([2, 0, 1] )
        return image, scale

    def loadMask(self, imName):
        if not osp.isfile(imName ):
            print('Error: %s does not exist.' % imName )
            assert(False )
        image = cv2.imread(imName, -1 )
        image = cv2.resize(image, (self.imWidth, self.imHeight ), interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image )[np.newaxis, :, :]
        return image

    def loadImage(self, imName, isGama = False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            assert(False)
            return im

        im = Image.open(imName)
        im = self.imResize(im)
        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            #im = 2 * im - 1
        else:
            #im = (im - 127.5) / 127.5
            im = (im - 0) / 255
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        return im

    def imResize(self, im):
        w0, h0 = im.size
        if w0 != self.imHeight or h0 != self.imWidth:
            im = im.resize( (self.imWidth, self.imHeight ), Image.ANTIALIAS)
        return im


#-------------------------------------------------------------------------------------------------------------
#创建真实模型在虚拟环境中的数据集，batchloader 仅读取相机、环境图、gt模型
class BatchLoaderMyreal(Dataset):
    def __init__(self, dataRoot, shapeRoot = None,
            imHeight = 360, imWidth = 480,
            envHeight = 256, envWidth = 512,
            isRandom=False, phase='TRAIN', rseed = 1,
            isLoadVH = False, isLoadEnvmap = False,
            isLoadCam = False, isLoadOptim = False,
            camNum = 10, shapeRs = 0, shapeRe = 1500, volumeSize=32, batchSize = None, isOptim = False, ignore = [],
                 isLoadSDF = True, grid_res = 8, bounding_radius = 1.1):

        self.dataRoot = dataRoot
        self.shapeRoot = shapeRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.phase = phase.upper()
        self.isLoadVH = isLoadVH
        self.isLoadCam = isLoadCam
        self.isLoadEnvmap = isLoadEnvmap
        self.camNum = camNum
        self.shapeRs = shapeRs
        self.shapeRe = shapeRe
        self.isLoadSDF = isLoadSDF
        self.grid_res = grid_res
        self.bounding_radius = bounding_radius

        if batchSize is None:
            batchSize = camNum
            self.batchSize = min(batchSize , 10)
        else:
            self.batchSize = batchSize

        self.minX, self.maxX = -bounding_radius, bounding_radius
        self.minY, self.maxY = -bounding_radius, bounding_radius
        self.minZ, self.maxZ = -bounding_radius, bounding_radius
        self.volumeSize = volumeSize
        y, x, z = np.meshgrid(
                np.linspace(self.minX, self.maxX, volumeSize ),
                np.linspace(self.minY, self.maxY, volumeSize ),
                np.linspace(self.minZ, self.maxZ, volumeSize ) )
        x = x[:, :, :, np.newaxis ]
        y = y[:, :, :, np.newaxis ]
        z = z[:, :, :, np.newaxis ]
        coord = np.concatenate([x, y, z], axis=3 )

        shapeList = sorted(glob.glob(osp.join(dataRoot, 'Shape__*') ))
        if isLoadCam:
            self.originArr = []
            self.lookatArr = []
            self.upArr = []
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
                if n in ignore:
                    continue
                shape = osp.join(shapeRoot, 'Shape__%d' % n )
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
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
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

                    envFileName = envFileName.replace('/home/zhl/CVPR20/TransparentShape','../')
                    if not osp.isfile(envFileName):
                        print(shapeList[n])
                    if not envFileName.find('1640')==-1:
                        print(shapeList[n])
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
        imScale = None
        shapeList = glob.glob(osp.join(self.dataRoot, 'Shape__*'))
        shapeId = ind
        batchDict = {}
        batchDict['data_path'] = osp.join(self.dataRoot, "Shape__%d" % (shapeId + self.shapeRs))
        for imId in self.perm:
            if self.isLoadCam:
                origin = self.originArr[shapeId ][imId ]
                lookat = self.lookatArr[shapeId ][imId ]
                up = self.upArr[shapeId ][imId ]

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
            sdfName = osp.join(shapePath, 'visualHullSubd_%d_%d_sdf.npy' % (self.camNum,self.grid_res))
            batchDict['shape_path'] = shapePath

            gt_sdfName = osp.join(shapePath, 'object_sdf_%d.npy'%(self.grid_res))
            if osp.isfile(gt_sdfName):
                batchDict['gt_grid'] = np.load(gt_sdfName).astype(np.float)
            else:
                #gtName = osp.join(shapePath, 'meshGT_transform.ply')
                gtName = osp.join(shapePath, 'object-1500000.obj')
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
        #image = np.clip((image * scale), 0, 1).transpose([2, 0, 1])
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

#-----------------------------------------------------------------------------------------------------------
#自己创建的真实模型虚拟环境数据集
class BatchLoaderMyReal(Dataset):
    def __init__(self, dataRoot, shapeRoot = None,
            imHeight = 360, imWidth = 480,
            envHeight = 256, envWidth = 512,
            isRandom=False, phase='TRAIN', rseed = 1,
            isLoadVH = False, isLoadEnvmap = False,
            isLoadCam = False, isLoadOptim = False,
            camNum = 10, shapeRs = 0, shapeRe = 1500, volumeSize=32, batchSize = None, isOptim = False, ignore = [],
                 isLoadSDF = True, grid_res = 8, bounding_radius = 1.1):

        self.dataRoot = dataRoot
        self.shapeRoot = shapeRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.phase = phase.upper()
        self.isLoadVH = isLoadVH
        self.isLoadCam = isLoadCam
        self.isLoadEnvmap = isLoadEnvmap
        self.isLoadOptim = isLoadOptim
        self.camNum = camNum
        self.shapeRs = shapeRs
        self.shapeRe = shapeRe
        self.isOptim = isOptim
        self.isLoadSDF = isLoadSDF
        self.grid_res = grid_res
        self.bounding_radius = bounding_radius

        if batchSize is None:
            batchSize = camNum
            self.batchSize = min(batchSize , 10)
        else:
            self.batchSize = batchSize

        self.minX, self.maxX = -1.1, 1.1
        self.minY, self.maxY = -1.1, 1.1
        self.minZ, self.maxZ = -1.1, 1.1
        self.volumeSize = volumeSize
        y, x, z = np.meshgrid(
                np.linspace(self.minX, self.maxX, volumeSize ),
                np.linspace(self.minY, self.maxY, volumeSize ),
                np.linspace(self.minZ, self.maxZ, volumeSize ) )
        x = x[:, :, :, np.newaxis ]
        y = y[:, :, :, np.newaxis ]
        z = z[:, :, :, np.newaxis ]
        coord = np.concatenate([x, y, z], axis=3 )

        shapeList = glob.glob(osp.join(dataRoot, 'Shape__*') )
        if isLoadCam:
            self.originArr = []
            self.lookatArr = []
            self.upArr = []
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
                if n in ignore:
                    continue
                shape = osp.join(shapeRoot, 'Shape__%d' % n )
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
            for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
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

                    envFileName = envFileName.replace('/home/zhl/CVPR20/TransparentShape','../')
                    if not osp.isfile(envFileName):
                        print(shapeList[n])
                    if not envFileName.find('1640')==-1:
                        print(shapeList[n])
                    floats = shape.findall('float')
                    assert(len(floats) == 1 )
                    for f in floats:
                        scale = float(f.get('value') )
                    self.envList.append(envFileName )
                    self.scaleList.append(scale )

                    if envFileName not in envListUnique:
                        envListUnique.append(envFileName )
            print("Number of environment maps %d" % (len(envListUnique ) ) )

        self.imList = []
        for n in range(max(0, shapeRs ), min(len(shapeList ), shapeRe ) ):
            if n in ignore:
                continue
            shape = osp.join(dataRoot, 'Shape__%d' % n )
            if not osp.isdir(shape ):
                continue
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.npy' ) ) )
            if isRandom:
                random.shuffle(imNames )
            if len(imNames ) < camNum:
                print('%s: %d' % (shape, len(imNames) ) )
                assert(False )
            self.imList.append(imNames[0:camNum ] )

        if rseed is not None:
            random.seed(rseed)

        # Permute the image list
        self.count = len(self.imList)
        self.perm = list(range(self.count ) )
        if isRandom:
            random.shuffle(self.perm)




    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        # normalize the normal vector so that it will be unit length

        imNames = self.imList[self.perm[ind ] ]
        if self.batchSize < self.camNum:
            random.shuffle(imNames )

            #当batchsize 为 camnum - 1时，最后一个照片替换为法向量camNum
            if self.isOptim:
                count = 0
                imNamesNew = []
                for n in range(0, self.camNum ):
                    isAdd = False
                    imName = imNames[n]
                    if n == self.camNum-2:
                        isAdd = True
                    else:
                        twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) )
                        if not osp.isfile(twoNormalName ):
                            isAdd = True
                    if isAdd == True:
                        imNamesNew.append(imName )
                        count += 1
                    if count == self.batchSize:
                        break
                imNames = imNamesNew
            else:
                imNames = imNames[0:self.batchSize ]

        segs = []
        seg2s = []
        normals = []
        normal2s = []
        depths = []
        depth2s = []
        ims = []
        imEs = []

        origins = []
        lookats = []
        ups = []

        envs = []

        segVHs = []
        seg2VHs = []
        normalVHs = []
        normal2VHs = []
        depthVHs = []
        depth2VHs = []

        normalOpts = []
        normal2Opts = []

        imScale = None
        for imName in imNames:
            twoBounceName = imName.replace('im_', 'imtwoBounce_')
            if not osp.isfile(twoBounceName ):
                twoBounceName = imName.replace('im_', 'imtwoBounce_').replace('.npy', '.h5')
                hf = h5py.File(twoBounceName, 'r')
                twoBounce = np.array(hf.get('data'), dtype=np.float32 )
                hf.close()
            else:
                twoBounce = np.load(twoBounceName )

            if twoBounce.shape[0] != self.imWidth or twoBounce.shape[1] != self.imHeight:
                newTwoBounce1 = cv2.resize(twoBounce[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce2 = cv2.resize(twoBounce[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce3 = cv2.resize(twoBounce[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                newTwoBounce4 = cv2.resize(twoBounce[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce5 = cv2.resize(twoBounce[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce6 = cv2.resize(twoBounce[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                twoBounce = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                    newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

            normal = twoBounce[:, :, 0:3].transpose([2, 0, 1] )
            normal = np.ascontiguousarray(normal )
            seg = twoBounce[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
            seg = np.ascontiguousarray(seg.astype(np.float32) )
            #seg = seg[:,:,::-1]

            depth = twoBounce[:, :, 3:6].transpose([2, 0, 1] )
            depth = np.ascontiguousarray(depth )
            depth = depth * seg

            normal2 = twoBounce[:, :, 7:10].transpose([2, 0, 1] )
            normal2 = np.ascontiguousarray(normal2 )
            seg2 = twoBounce[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
            seg2 = np.ascontiguousarray(seg2.astype(np.float32) )

            depth2 = twoBounce[:, :, 10:13].transpose([2, 0, 1] )
            depth2 = np.ascontiguousarray(depth2 )
            depth2 = depth2 * seg

            normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-10) )[np.newaxis, :]
            normal = normal * seg
            normal2 = normal2 / np.sqrt(np.maximum(np.sum(normal2 * normal2, axis=0), 1e-10) )[np.newaxis, :]
            normal2 = normal2 * seg


            # Read rendered images(照片value压缩到0~1)
            imE, imScale = self.loadHDR(imName, imScale )
            #imE = imE[:,:,::-1]
            im = imE * seg

            imId = int(imName.split('/')[-1].split('.')[0].split('_')[-1]  )
            shapeId = int(imName.split('/')[-2].split('_')[-1] ) - self.shapeRs

            segs.append(seg[np.newaxis, :] )
            seg2s.append(seg2[np.newaxis, :] )
            normals.append(normal[np.newaxis, :] )
            normal2s.append(normal2[np.newaxis, :] )
            depths.append(depth[np.newaxis, :] )
            depth2s.append(depth2[np.newaxis, :] )
            ims.append(im[np.newaxis, :] )
            imEs.append(imE[np.newaxis, :] )

            # Load the rendering file
            if self.isLoadCam:
                origin = self.originArr[shapeId ][imId-1 ]
                lookat = self.lookatArr[shapeId ][imId-1 ]
                up = self.upArr[shapeId ][imId-1 ]

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
                env = env.transpose([2, 0, 1]) * imScale * scale

                envs.append(env[np.newaxis, :] )


            if self.isLoadVH:
                twoBounceVHName = imName.replace('im_', 'imVH_%dtwoBounce_' % self.camNum )
                if not osp.isfile(twoBounceVHName ):
                    twoBounceVHName = imName.replace('im_', 'imVH_%dtwoBounce_' % self.camNum ).replace('.npy', '.h5')
                    hf = h5py.File(twoBounceVHName, 'r')
                    twoBounceVH = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoBounceVH = np.load(twoBounceVHName )

                if twoBounceVH.shape[0] != self.imWidth or twoBounceVH.shape[1] != self.imHeight:
                    newTwoBounce1 = cv2.resize(twoBounceVH[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce2 = cv2.resize(twoBounceVH[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce3 = cv2.resize(twoBounceVH[:, :, 6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    newTwoBounce4 = cv2.resize(twoBounceVH[:, :, 7:10], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce5 = cv2.resize(twoBounceVH[:, :, 10:13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                    newTwoBounce6 = cv2.resize(twoBounceVH[:, :, 13], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                    twoBounceVH = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce3[:, :, np.newaxis],
                        newTwoBounce4, newTwoBounce5, newTwoBounce6[:, :, np.newaxis] ), axis=2)

                normalVH = twoBounceVH[:, :, 0:3].transpose([2, 0, 1])
                normalVH = np.ascontiguousarray(normalVH )
                segVH = twoBounceVH[:, :, 6:7].transpose([2, 0, 1] ) > 0.9
                segVH = np.ascontiguousarray(segVH.astype(np.float32) )

                depthVH = twoBounceVH[:, :, 3:6].transpose([2, 0, 1])
                depthVH = np.ascontiguousarray(depthVH )
                depthVH = depthVH * segVH

                normal2VH = twoBounceVH[:, :, 7:10].transpose([2, 0, 1])
                normal2VH = np.ascontiguousarray(normal2VH )
                seg2VH = twoBounceVH[:, :, 13:14].transpose([2, 0, 1] ) > 0.9
                seg2VH = np.ascontiguousarray(seg2VH.astype(np.float32) )

                depth2VH = twoBounceVH[:, :, 10:13].transpose([2, 0, 1])
                depth2VH = np.ascontiguousarray(depth2VH )
                depth2VH = depth2VH * segVH

                normalVH = normalVH / np.sqrt(np.maximum(np.sum(normalVH * normalVH, axis=0), 1e-10) )[np.newaxis, :]
                normalVH = normalVH * segVH
                normal2VH = normal2VH / np.sqrt(np.maximum(np.sum(normal2VH * normal2VH, axis=0), 1e-10) )[np.newaxis, :]
                normal2VH = normal2VH * segVH

                segVHs.append(segVH[np.newaxis, :] )
                seg2VHs.append(seg2VH[np.newaxis, :] )
                normalVHs.append(normalVH[np.newaxis, :] )
                normal2VHs.append(normal2VH[np.newaxis, :] )
                depthVHs.append(depthVH[np.newaxis, :] )
                depth2VHs.append(depth2VH[np.newaxis, :] )

            if self.isLoadOptim:
                twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) )
                if not osp.isfile(twoNormalName ):
                    twoNormalName = imName.replace('im_', 'imtwoNormalPred%d_' % (self.camNum ) ).replace('.npy', '.h5')
                    hf = h5py.File(twoNormalName, 'r')
                    twoNormals = np.array(hf.get('data'), dtype=np.float32 )
                    hf.close()
                else:
                    twoNormals = np.load(twoNormalName )
                normalOpt, normal2Opt = twoNormals[:, :, 0:3], twoNormals[:, :, 3:6]
                normalOpt = cv2.resize(normalOpt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normal2Opt = cv2.resize(normal2Opt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
                normalOpt = np.ascontiguousarray(normalOpt.transpose([2, 0, 1] ) )
                normal2Opt = np.ascontiguousarray(normal2Opt.transpose([2, 0, 1] ) )

                normalOpt = normalOpt / np.sqrt(np.maximum(np.sum(normalOpt * normalOpt, axis=0), 1e-10) )[np.newaxis, :]
                normalOpt = normalOpt * seg
                normal2Opt = normal2Opt / np.sqrt(np.maximum(np.sum(normal2Opt * normal2Opt, axis=0), 1e-10) )[np.newaxis, :]
                normal2Opt = normal2Opt * seg

                normalOpts.append(normalOpt[np.newaxis, :] )
                normal2Opts.append(normal2Opt[np.newaxis, :] )

        segs = np.concatenate(segs, axis=0 )
        seg2s = np.concatenate(seg2s, axis=0 )
        normals = np.concatenate(normals, axis=0 )
        normal2s = np.concatenate(normal2s, axis=0 )
        depths = np.concatenate(depths, axis=0 )
        depth2s = np.concatenate(depth2s, axis=0 )
        ims = np.concatenate(ims, axis=0 )
        imEs = np.concatenate(imEs, axis=0 )

        batchDict = {'seg1': segs,
                'seg2': seg2s,
                'normal1': normals,
                'normal2': normal2s,
                'depth1': depths,
                'depth2': depth2s,
                'im':  ims,
                'imE': imEs,
                'name': imNames }

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

        if self.isLoadVH:
            segVHs = np.concatenate(segVHs, axis=0 )
            seg2VHs = np.concatenate(seg2VHs, axis=0 )
            normalVHs = np.concatenate(normalVHs, axis=0 )
            normal2VHs = np.concatenate(normal2VHs, axis=0 )
            depthVHs = np.concatenate(depthVHs, axis=0 )
            depth2VHs = np.concatenate(depth2VHs, axis=0 )

            batchDict['seg1VH'] = segVHs
            batchDict['seg2VH'] = seg2VHs
            batchDict['normal1VH'] = normalVHs
            batchDict['normal2VH'] = normal2VHs
            batchDict['depth1VH'] = depthVHs
            batchDict['depth2VH'] = depth2VHs

        if self.isLoadOptim:
            normalOpts = np.concatenate(normalOpts, axis=0 )
            normal2Opts = np.concatenate(normal2Opts, axis=0 )
            batchDict['normalOpt'] = normalOpts
            batchDict['normal2Opt'] = normal2Opts

        #读取sdf文件
        if self.isLoadSDF:
            imName = imNames[0]
            shapeId = imName.split('/')[-2]
            shapePath = osp.join(self.shapeRoot, shapeId)
            sdfName = osp.join(shapePath, 'visualHullSubd_%d_%d_sdf.npy' % (self.camNum,self.grid_res))
            batchDict['shape_path'] = shapePath
            if osp.isfile(sdfName):
                batchDict['grid'] = np.load(sdfName).astype(np.float)
            else:
                VHName = osp.join(shapePath, 'visualHullSubd_%d.obj' % self.camNum)
                mesh = trimesh.load(VHName)
                linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, self.grid_res)
                grid_x, grid_y, grid_z = np.meshgrid(linear_space, linear_space, linear_space)
                coords = np.stack((grid_x, grid_y, grid_z),axis=3)
                query_points = coords.reshape((-1,3))
                sdfs = mesh_to_sdf(mesh, query_points, surface_point_method='sample', sign_method='normal',
                                   bounding_radius=None, scan_count=100,
                                   scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
                sdfs = np.reshape(sdfs, grid_x.shape).transpose((1,0,2))
                batchDict['grid'] = sdfs
                np.save(sdfName,sdfs)

            grid_ress = [self.grid_res]
            for grid_res in grid_ress:
                gt_sdfName = osp.join(shapePath, 'object_sdf_%d.npy'%(grid_res))
                if osp.isfile(gt_sdfName):
                    batchDict['gt_grid'] = np.load(gt_sdfName).astype(np.float)
                else:
                    #gtName = osp.join(shapePath, 'meshGT_transform.ply')
                    gtName = osp.join(shapePath, 'object-1500000.obj')
                    gtmesh = trimesh.load(gtName)
                    linear_space = np.linspace(-self.bounding_radius, self.bounding_radius, grid_res)
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
        image = np.load(imName)[:, :, :]
        image = cv2.resize(image, (self.imWidth, self.imHeight ), interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image )
        imMean = np.mean(image )

        if scale is None:
            if self.phase == 'TRAIN':
                scale = (np.random.random() * 0.2 + 0.4) / imMean
            else:
                scale = 0.5 / imMean
        image = (image*scale).transpose([2, 0, 1] )
        #image = np.clip((image * scale), 0, 1).transpose([2, 0, 1])
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