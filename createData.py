from __future__ import print_function
import torch
import math
import argparse
import torchvision
import numpy as np
import random
from torch.autograd import Variable
import renderer
import time
import sys, os
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import dataLoader
from torch.utils.data import DataLoader
from utils import *
import pytorch_ssim
from mesh import create_mesh, convert_sdf_samples_to_ply, GenMeshfromSDF, CalChamferDis
import logging
import os.path as osp
from pytorch3d.renderer import cameras
from pytorch3d.io import load_obj
import cv2
import h5py
from torchvision import transforms
import imageio


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='../Data/Images%d/myreal', help='path to images')
parser.add_argument('--shapeRoot', default='../Data/Shapes/myreal/', help='path to images')
parser.add_argument('--experiment', default='../Data/Results/test/', help='the path to store samples and models')
parser.add_argument('--testRoot', default=None, help='the path to store outputs')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=None, help='input batch size')
parser.add_argument('--imageHeight', type=int, default=192, help='the height / width of the input image to network')
parser.add_argument('--imageWidth', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--envHeight', type=int, default=1024, help='the height / width of the input envmap to network')
parser.add_argument('--envWidth', type=int, default=2048, help='the height / width of the input envmap to network')
# The parameters
parser.add_argument('--camNum', type=int, default=10, help='the number of views to create the visual hull')
parser.add_argument('--sampleNum', type=int, default=1, help='the sample num for the cost volume')
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape')
parser.add_argument('--shapeEnd', type=int, default=1, help='the end id of the shape')
parser.add_argument('--isAddCostVolume', action='store_true', help='whether to use cost volume or not')
parser.add_argument('--poolingMode', type=int, default=2, help='0: maxpooling, 1: average pooling 2: learnable pooling')
parser.add_argument('--isNoErrMap', action='store_true', help='whether to remove the error map in the input')
# The rendering parameters
parser.add_argument('--eta1', type=float, default=1.0003, help='the index of refraction of air')
parser.add_argument('--eta2', type=float, default=1.4723, help='the index of refraction of glass')
parser.add_argument('--fov', type=float, default=63.23, help='the x-direction full field of view of camera')
# The loss parameters
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for normal')
# The gpu setting
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--ball', type=float, default=0, help='whether to use ball as initia shape')
parser.add_argument('--gridsize', type=int, default=256, help='learning rate')
parser.add_argument('--maskloss', type=float, default=1, help='whether to use mask loss')

torch.set_printoptions(precision=8)
opt = parser.parse_args()
print(opt)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_name = opt.experiment
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
if not os.path.isfile(opt.experiment + 'log.txt'):
    os.mknod(opt.experiment + 'log.txt')
# logging.basicConfig(level=logging.WARNING,
#                     filename= opt.experiment + 'log.txt',
#                     filemode='w',
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
fhlr = logging.FileHandler(opt.experiment + 'log.txt')
fhlr.setFormatter(formatter)
logger.addHandler(fhlr)

logger.info(opt)

opt.gpuId = opt.deviceIds[0]
opt.dataRoot = opt.dataRoot % opt.camNum

nw = opt.normalWeight
if opt.batchSize is None:
    opt.batchSize = opt.camNum

if __name__ == "__main__":



    brdfDataset = dataLoader.BatchLoaderMyreal(
        opt.dataRoot, shapeRoot=opt.shapeRoot,
        imHeight=opt.imageHeight, imWidth=opt.imageWidth,
        envHeight=opt.envHeight, envWidth=opt.envWidth,
        isRandom=False, phase='TEST', rseed=1,
        isLoadVH=True, isLoadEnvmap=True, isLoadCam=True,
        shapeRs=opt.shapeStart, shapeRe=opt.shapeEnd,
        camNum=opt.camNum, batchSize=opt.camNum, isLoadSDF=True, grid_res=opt.gridsize, bounding_radius=1.1)

    brdfLoader = DataLoader(brdfDataset, batch_size=1, num_workers=1, shuffle=False)

    j = 0

    normal1ErrsNpList = np.ones([1, 2], dtype=np.float32)
    meanAngle1ErrsNpList = np.ones([1, 2], dtype=np.float32)
    medianAngle1ErrsNpList = np.ones([1, 2], dtype=np.float32)
    normal2ErrsNpList = np.ones([1, 2], dtype=np.float32)
    meanAngle2ErrsNpList = np.ones([1, 2], dtype=np.float32)
    medianAngle2ErrsNpList = np.ones([1, 2], dtype=np.float32)
    renderedErrsNpList = np.ones([1, 2], dtype=np.float32)

    epoch = opt.nepoch
    # testingLog = open('{0}/testingLog_{1}.txt'.format(opt.testRoot, epoch), 'w')
    for i, dataBatch in enumerate(brdfLoader):
        j += 1
        # Load environment map
        envmap_cpu = dataBatch['env'].squeeze(0)
        envBatch = Variable(envmap_cpu).cuda()

        # Load camera parameters
        origin_cpu = dataBatch['origin'].squeeze(0)
        originBatch = Variable(origin_cpu).cuda()

        lookat_cpu = dataBatch['lookat'].squeeze(0)
        lookatBatch = Variable(lookat_cpu).cuda()

        up_cpu = dataBatch['up'].squeeze(0)
        upBatch = Variable(up_cpu).cuda()

        # Load visual hull data
        grid_gt_cpu = dataBatch['gt_grid'].squeeze(0)
        grid_gt = Variable(grid_gt_cpu).cuda()

        shapePath = dataBatch['shape_path'][0]
        dataPath = dataBatch['data_path'][0]
        batchSize = originBatch.size(0)

        # ---------------------------------------------------------------------------------------------------------------
        # define the folder name for results
        channelNum = envBatch.size(1)
        # Speed up
        # torch.backends.cudnn.benchmark = True
        R, T = cameras.look_at_view_transform(eye=originBatch.cpu().numpy(), up=upBatch.cpu().numpy(),
                                              at=lookatBatch.cpu().numpy(), device=device)
        focal_length_pixel = opt.imageWidth / 2 / (np.tan(opt.fov / 2 / 180 * np.pi))
        cameras_bs = cameras.PerspectiveCameras(focal_length=focal_length_pixel,
                                                principal_point=np.tile(np.array((opt.imageWidth / 2, opt.imageHeight / 2)),
                                                                        (opt.camNum, 1)), R=R, T=T, device=device,
                                                image_size=np.tile(np.array((opt.imageWidth, opt.imageHeight)),
                                                                        (opt.camNum, 1)))
        bs_coord = torch.linspace(0,opt.camNum - 1,opt.camNum).reshape(opt.camNum,1,1).repeat((1,opt.imageHeight,opt.imageWidth)).long()

        cuda = True if torch.cuda.is_available() else False
        print(cuda)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        width = opt.imageWidth
        height = opt.imageHeight

        # bounding box
        bounding_box_min_x = -1.1
        bounding_box_min_y = -1.1
        bounding_box_min_z = -1.1
        bounding_box_max_x = 1.1
        bounding_box_max_y = 1.1
        bounding_box_max_z = 1.1

        # initialize the grid
        # define the resolutions of the multi-resolution part
        voxel_res_list = [32, 40, 48, 56, 64, 128, 256]
        # grid_res_x = grid_res_y = grid_res_z = voxel_res_list.pop(0)
        grid_res_x = grid_res_y = grid_res_z = opt.gridsize
        voxel_size = Tensor([(bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1)])

        # Construct the sdf grid
        if (opt.ball):
            grid_initial = grid_construction_sphere_small(grid_res_x, bounding_box_min_x, bounding_box_max_x,
                                                          opt.ball)  ####
        else:
            grid_initial = grid_gt.float()
        # set parameters
        sdf_diff_list = []
        time_list = []
        image_loss = 1000 * batchSize
        sdf_loss = 1000 * batchSize
        iterations = 0
        scale = 1
        start_time = time.time()
        learning_rate = opt.lr
        tolerance = 1e-8

        image_initial, attmask, mask, inter_min_index, fine_pos = generate_image_bs(bounding_box_min_x,
                                                                                    bounding_box_min_y,
                                                                                    bounding_box_min_z,
                                                                                    bounding_box_max_x,
                                                                                    bounding_box_max_y,
                                                                                    bounding_box_max_z,
                                                                                    voxel_size,
                                                                                    grid_res_x, grid_res_y, grid_res_z,
                                                                                    batchSize, width,
                                                                                    height,
                                                                                    grid_initial,
                                                                                    opt.fov / 2, originBatch,
                                                                                    lookatBatch, upBatch,
                                                                                    opt.eta1,
                                                                                    opt.eta2,
                                                                                    envBatch.permute(0, 2, 3, 1),
                                                                                    opt.envHeight, opt.envWidth)
        image_np = image_initial.cpu().numpy()
        mask_np = mask.cpu().numpy()
        image_initial = image_initial.permute(0, 3, 1, 2)

        images = np.split(image_np,opt.camNum)
        masks = np.split(mask_np,opt.camNum)
        for i in range(opt.camNum):
            curimage = images[i].squeeze(0)
            curmask = np.tile(masks[i].squeeze(0),(1,1,14))

            np.save(osp.join(dataPath,'im_%d.npy' % (i + 1)),curimage)
            np.save(osp.join(dataPath,'imtwoBounce_%d.npy' % (i + 1)),curmask)
            np.save(osp.join(dataPath, 'imVH_%dtwoBounce_%d.npy' % (opt.camNum,i + 1)), curmask)


        print("Time:", time.time() - start_time)

        print("----- END -----")
