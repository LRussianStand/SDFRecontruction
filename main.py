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

from torchvision import transforms

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='../Data/Images%d/test', help='path to images')
parser.add_argument('--shapeRoot', default='../Data/Shapes/test/', help='path to images')
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
parser.add_argument('--fov', type=float, default=63.4, help='the field of view of camera')
# The loss parameters
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for normal')
# The gpu setting
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
opt.dataRoot = opt.dataRoot % opt.camNum

nw = opt.normalWeight
if opt.batchSize is None:
    opt.batchSize = opt.camNum

if __name__ == "__main__":

    brdfDataset = dataLoader.BatchLoader(
        opt.dataRoot, shapeRoot=opt.shapeRoot,
        imHeight=opt.imageHeight, imWidth=opt.imageWidth,
        envHeight=opt.envHeight, envWidth=opt.envWidth,
        isRandom=False, phase='TEST', rseed=1,
        isLoadVH=True, isLoadEnvmap=True, isLoadCam=True,
        shapeRs=opt.shapeStart, shapeRe=opt.shapeEnd,
        camNum=opt.camNum, batchSize=opt.camNum, isLoadSDF=True, grid_res= 256, bounding_radius= 1.1)
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
        # Load ground-truth from cpu to gpu
        normal1_cpu = dataBatch['normal1'].squeeze(0)
        normal1Batch = Variable(normal1_cpu).cuda()

        seg1_cpu = dataBatch['seg1'].squeeze(0)
        seg1Batch = Variable((seg1_cpu)).cuda()

        normal2_cpu = dataBatch['normal2'].squeeze(0)
        normal2Batch = Variable(normal2_cpu).cuda()

        seg2_cpu = dataBatch['seg2'].squeeze(0)
        seg2Batch = Variable(seg2_cpu).cuda()

        # Load the image from cpu to gpu
        im_cpu = dataBatch['im'].squeeze(0)
        imBatch = Variable(im_cpu).cuda()

        imBg_cpu = dataBatch['imE'].squeeze(0)
        imBgBatch = Variable(imBg_cpu).cuda()

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
        normal1VH_cpu = dataBatch['normal1VH'].squeeze(0)
        normal1VHBatch = Variable(normal1VH_cpu).cuda()

        seg1VH_cpu = dataBatch['seg1VH'].squeeze(0)
        seg1VHBatch = Variable((seg1VH_cpu)).cuda()

        normal2VH_cpu = dataBatch['normal2VH'].squeeze(0)
        normal2VHBatch = Variable(normal2VH_cpu).cuda()

        seg2VH_cpu = dataBatch['seg2VH'].squeeze(0)
        seg2VHBatch = Variable(seg2VH_cpu).cuda()

        grid_vh_cpu = dataBatch['grid'].squeeze(0)
        grid_vh = Variable(grid_vh_cpu).cuda()

        grid_gt_cpu = dataBatch['gt_grid'].squeeze(0)
        grid_gt = Variable(grid_gt_cpu).cuda()

        shapePath = dataBatch['shape_path'].squeeze(0)

        batchSize = normal1Batch.size(0)

        # ---------------------------------------------------------------------------------------------------------------
        # define the folder name for results
        dir_name = opt.experiment
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # Speed up
        # torch.backends.cudnn.benchmark = True

        cuda = True if torch.cuda.is_available() else False
        print(cuda)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        width = opt.imageWidth
        height = opt.imageHeight

        # cameras
        listOrigin = torch.split(originBatch, 1, 0)
        listLookat = torch.split(lookatBatch, 1, 0)
        listUp = torch.split(upBatch, 1, 0)
        # bounding box
        bounding_box_min_x = -1.1
        bounding_box_min_y = -1.1
        bounding_box_min_z = -1.1
        bounding_box_max_x = 1.1
        bounding_box_max_y = 1.1
        bounding_box_max_z = 1.1

        # images
        listEnv = torch.split(envBatch, 1, 0)
        listImg = torch.split(imBgBatch, 1, 0)

        # initialize the grid
        # define the resolutions of the multi-resolution part
        voxel_res_list = [8, 16, 24, 32, 40, 48, 56, 64, 128,256]
        #grid_res_x = grid_res_y = grid_res_z = voxel_res_list.pop(0)
        grid_res_x = grid_res_y = grid_res_z = 256
        voxel_size = Tensor([(bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1)])

        # Construct the sdf grid
        #grid_initial = grid_construction_sphere_small(grid_res_x, bounding_box_min_x, bounding_box_max_x)  ####
        grid_initial = grid_vh.float()
        # set parameters
        sdf_diff_list = []
        time_list = []
        image_loss = [1000] * len(listOrigin)
        sdf_loss = [1000] * len(listOrigin)
        iterations = 0
        scale = 1
        start_time = time.time()
        learning_rate = opt.lr
        tolerance = 0.4

        # train
        start_time = time.time()
        while (grid_res_x <= 256):
            # tolerance *= 1.05
            image_target = []
            grid_initial.requires_grad = True
            optimizer = torch.optim.Adam([grid_initial], lr=learning_rate, eps=1e-2)

            # output initial images
            for m in range(len(listOrigin)):
                refractImg, reflectImg, attmask, mask = generate_image(bounding_box_min_x, bounding_box_min_y,
                                                                 bounding_box_min_z,
                                                                 bounding_box_max_x, bounding_box_max_y,
                                                                 bounding_box_max_z,
                                                                 voxel_size,
                                                                 grid_res_x, grid_res_y, grid_res_z, width, height,
                                                                 grid_initial,
                                                                 opt.fov / 2, listOrigin[m].squeeze(0),
                                                                 listLookat[m].squeeze(0), listUp[m].squeeze(0),
                                                                 opt.eta1,
                                                                 opt.eta2, listEnv[m].squeeze(0).permute(1, 2, 0),
                                                                 opt.envHeight, opt.envWidth)
                #image_initial = (torch.clamp(refractImg + reflectImg, 0, 1)).data.permute(2, 0, 1)
                image_initial = (torch.clamp(refractImg + reflectImg, 0, 1)).data.permute(0, 3, 1, 2)
                # image_initial = generate_image_lam(bounding_box_min_x, bounding_box_min_y,
                #                                                   bounding_box_min_z,
                #                                                   bounding_box_max_x, bounding_box_max_y,
                #                                                   bounding_box_max_z,
                #                                                   voxel_size,
                #                                                   grid_res_x, grid_res_y, grid_res_z, width, height,
                #                                                   grid_target,
                #                                                   opt.fov / 2, listOrigin[m].squeeze(0),
                #                                                   listLookat[m].squeeze(0), listUp[m].squeeze(0))
                torchvision.utils.save_image(image_initial,
                                             "./" + dir_name + "grid_res_" + str(grid_res_x) + "_start_" + str(
                                                 m) + ".png", nrow=8, padding=2, normalize=False, range=None,
                                             scale_each=False, pad_value=0)
                torchvision.utils.save_image(listImg[m],
                                             "./" + dir_name + "grid_res_" + str(grid_res_x) + "_gt_" + str(
                                                 m) + ".png", nrow=8, padding=2, normalize=False, range=None,
                                             scale_each=False, pad_value=0)

            # deform initial SDf to target SDF
            i = 0
            loss_camera = [1000] * len(listOrigin)
            average = 100000
            grid_loss_start = torch.sum(torch.abs(grid_initial - grid_gt)).item()
            grid_loss_last = grid_loss_start
            while sum(loss_camera) < average - tolerance / 2:
                average = sum(loss_camera)
                for cam in range(len(listOrigin)):
                    loss = 100000
                    prev_loss = loss + 1
                    num = 0
                    while ((num < 5) and loss < prev_loss):
                        num += 1
                        prev_loss = loss
                        iterations += 1

                        optimizer.zero_grad()

                        # Generate images
                        refractImg, reflectImg, attmask, mask = generate_image(bounding_box_min_x, bounding_box_min_y,
                                                                         bounding_box_min_z,
                                                                         bounding_box_max_x, bounding_box_max_y,
                                                                         bounding_box_max_z,
                                                                         voxel_size,
                                                                         grid_res_x, grid_res_y, grid_res_z, width,
                                                                         height, grid_initial,
                                                                         opt.fov / 2, listOrigin[cam].squeeze(0),
                                                                         listLookat[cam].squeeze(0),
                                                                         listUp[cam].squeeze(0), opt.eta1,
                                                                         opt.eta2,
                                                                         listEnv[cam].squeeze(0).permute(1, 2, 0),
                                                                         opt.envHeight, opt.envWidth)
                        image_initial = (torch.clamp(refractImg + reflectImg, 0, 1)).data.permute(2, 0, 1)
                        # Perform backprobagation
                        # compute image loss and sdf loss
                        image_loss[cam], sdf_loss[cam] = loss_fn_ssim(image_initial, listImg[cam].squeeze(0), grid_initial,
                                                                 voxel_size, grid_res_x, grid_res_y, grid_res_z,
                                                                 width, height)

                        # compute laplacian loss
                        conv_input = (grid_initial).unsqueeze(0).unsqueeze(0)
                        conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                                                [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]])
                        Lp_loss = torch.sum(F.conv3d(conv_input, conv_filter) ** 2)/(grid_res_x*grid_res_y*grid_res_y)

                        # get total loss
                        loss = 20 * image_loss[cam] + sdf_loss[cam] + Lp_loss
                        image_loss[cam] = image_loss[cam] / len(listOrigin)
                        sdf_loss[cam] = sdf_loss[cam] / len(listOrigin)
                        loss_camera[cam] = image_loss[cam] + sdf_loss[cam]

                        # print out loss messages
                        print("grid res:", grid_res_x, "iteration:", i, "num:", num, "loss:", loss, "\ncamera:",
                              listOrigin[cam])

                        loss.backward()
                        optimizer.step()
                        grid_loss = torch.sum(torch.abs(grid_initial - grid_gt)).item()
                        print(Notify.INFO, "grid_loss: %f grid_loss_change: %f grid_loss_total_change: %f" % (
                        grid_loss, grid_loss - grid_loss_last, grid_loss - grid_loss_start), Notify.ENDC)
                        grid_loss_last = grid_loss
                i += 1


            # genetate result images
            for cam in range(len(listOrigin)):
                refractImg, reflectImg, attmask, mask = generate_image(bounding_box_min_x, bounding_box_min_y,
                                                                 bounding_box_min_z,
                                                                 bounding_box_max_x, bounding_box_max_y,
                                                                 bounding_box_max_z,
                                                                 voxel_size,
                                                                 grid_res_x, grid_res_y, grid_res_z, width,
                                                                 height, grid_initial,
                                                                 opt.fov / 2, listOrigin[cam].squeeze(0),
                                                                 listLookat[cam].squeeze(0),
                                                                 listUp[cam].squeeze(0), opt.eta1,
                                                                 opt.eta2,
                                                                 listEnv[cam].squeeze(0).permute(1, 2, 0),
                                                                 opt.envHeight, opt.envWidth)
                image_initial = (torch.clamp(refractImg + reflectImg, 0, 1)).data.permute(2, 0, 1)
                torchvision.utils.save_image(image_initial,
                                             "./" + dir_name + "final_cam_" + str(grid_res_x) + "_" + str(
                                                 cam) + ".png", nrow=8, padding=2, normalize=False, range=None,
                                             scale_each=False, pad_value=0)

            # Save the final SDF result
            with open("./" + dir_name + str(grid_res_x) + "_best_sdf_bunny.pt", 'wb') as f:
                torch.save(grid_initial, f)

                # moves on to the next resolution stage
            if grid_res_x < 256:
                grid_res_update_x = grid_res_update_y = grid_res_update_z = voxel_res_list.pop(0)
                voxel_size_update = (bounding_box_max_x - bounding_box_min_x) / (grid_res_update_x - 1)
                grid_initial_update = Tensor(grid_res_update_x, grid_res_update_y, grid_res_update_z)
                linear_space_x = torch.linspace(0, grid_res_update_x - 1, grid_res_update_x)
                linear_space_y = torch.linspace(0, grid_res_update_y - 1, grid_res_update_y)
                linear_space_z = torch.linspace(0, grid_res_update_z - 1, grid_res_update_z)
                first_loop = linear_space_x.repeat(grid_res_update_y * grid_res_update_z, 1).t().contiguous().view(
                    -1).unsqueeze_(1)
                second_loop = linear_space_y.repeat(grid_res_update_z, grid_res_update_x).t().contiguous().view(
                    -1).unsqueeze_(1)
                third_loop = linear_space_z.repeat(grid_res_update_x * grid_res_update_y).unsqueeze_(1)
                loop = torch.cat((first_loop, second_loop, third_loop), 1).cuda()
                min_x = Tensor([bounding_box_min_x]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
                min_y = Tensor([bounding_box_min_y]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
                min_z = Tensor([bounding_box_min_z]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
                bounding_min_matrix = torch.cat((min_x, min_y, min_z), 1)

                # Get the position of the grid points in the refined grid
                points = bounding_min_matrix + voxel_size_update * loop
                voxel_min_point_index_x = torch.floor((points[:, 0].unsqueeze_(1) - min_x) / voxel_size).clamp(
                    max=grid_res_x - 2)
                voxel_min_point_index_y = torch.floor((points[:, 1].unsqueeze_(1) - min_y) / voxel_size).clamp(
                    max=grid_res_y - 2)
                voxel_min_point_index_z = torch.floor((points[:, 2].unsqueeze_(1) - min_z) / voxel_size).clamp(
                    max=grid_res_z - 2)
                voxel_min_point_index = torch.cat(
                    (voxel_min_point_index_x, voxel_min_point_index_y, voxel_min_point_index_z), 1)
                voxel_min_point = bounding_min_matrix + voxel_min_point_index * voxel_size

                # Compute the sdf value of the grid points in the refined grid
                grid_initial_update = calculate_sdf_value(grid_initial, points, voxel_min_point,
                                                          voxel_min_point_index, voxel_size, grid_res_x, grid_res_y,
                                                          grid_res_z).view(grid_res_update_x, grid_res_update_y,
                                                                           grid_res_update_z)

                # Update the grid resolution for the refined sdf grid
                grid_res_x = grid_res_update_x
                grid_res_y = grid_res_update_y
                grid_res_z = grid_res_update_z

                # Update the voxel size for the refined sdf grid
                voxel_size = voxel_size_update

                # Update the sdf grid
                grid_initial = grid_initial_update.data

                # Double the size of the image
                if width < 256:
                    width = int(width * 2)
                    height = int(height * 2)
                learning_rate /= 1.03

        print("Time:", time.time() - start_time)

        print("----- END -----")
