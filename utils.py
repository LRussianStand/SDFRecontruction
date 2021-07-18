# -*- coding: utf-8 -*-
import paddle

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import renderer
import pytorch_ssim


cuda = True if torch.cuda.is_available() else False
print(cuda)
#
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def grid_construction_sphere_small(grid_res, bounding_box_min, bounding_box_max,radius):
    # Construct the sdf grid for a sphere with radius 1
    linear_space = paddle.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = paddle.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - radius
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
    X_1 = paddle.cat(
        (grid[1:, :, :], (3 * grid[n_x, :, :] - 3 * grid[n_x - 1, :, :] + grid[n_x - 2, :, :]).unsqueeze_(0)), 0)
    X_2 = paddle.cat(((-3 * grid[1, :, :] + 3 * grid[0, :, :] + grid[2, :, :]).unsqueeze_(0), grid[:n_x, :, :]), 0)
    grid_normal_x = (X_1 - X_2) / (2 * voxel_size)

    # y-axis normal vectors
    Y_1 = paddle.cat(
        (grid[:, 1:, :], (3 * grid[:, n_y, :] - 3 * grid[:, n_y - 1, :] + grid[:, n_y - 2, :]).unsqueeze_(1)), 1)
    Y_2 = paddle.cat(((-3 * grid[:, 1, :] + 3 * grid[:, 0, :] + grid[:, 2, :]).unsqueeze_(1), grid[:, :n_y, :]), 1)
    grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size)

    # z-axis normal vectors
    Z_1 = paddle.cat(
        (grid[:, :, 1:], (3 * grid[:, :, n_z] - 3 * grid[:, :, n_z - 1] + grid[:, :, n_z - 2]).unsqueeze_(2)), 2)
    Z_2 = paddle.cat(((-3 * grid[:, :, 1] + 3 * grid[:, :, 0] + grid[:, :, 2]).unsqueeze_(2), grid[:, :, :n_z]), 2)
    grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size)

    return [grid_normal_x, grid_normal_y, grid_normal_z]

def get_intersection_normal_bs(intersection_grid_normal, intersection_pos, voxel_min_point, voxel_size):
    # Compute parameters batchsize
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

def compute_intersection_pos_bs(grid, intersection_pos_rough, voxel_min_point, voxel_min_point_index, ray_direction,
                             voxel_size, mask,batchsize, width, height):
    # Linear interpolate along x axis the eight values (batch size)
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

#----------------------------------------------------------------------------------------------------------

def generate_image_bs(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                   bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                   voxel_size, grid_res_x, grid_res_y, grid_res_z, batchsize, width, height, grid, fov, origin, lookat, up, eta1,
                   eta2,envmap,envHeight,envWidth):
    b_w_h_3 = paddle.zeros(batchsize,height, width, 3).cuda()
    b_w_h = paddle.zeros(batchsize, height, width).cuda()
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


    #------------------------------------------------------------------------------------------------------------------------
    dis2box_xyz = paddle.ones_like(mask) * math.sqrt(3) * (bounding_box_max_x - bounding_box_min_x)
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

    maskatt = paddle.clip((mask1 + mask2 + mask3 + mask4).float(), min=0, max=1)

    return (refractImg1 + reflectImg + refractImg2 + refractImg3),maskatt, mask, voxel_min_point_index,intersection_pos1


# The energy E captures the difference between a rendered image and
# a desired target image, and the rendered image is a function of the
# SDF values. You could write E(SDF) = ||rendering(SDF)-target_image||^2.
# In addition, there is a second term in the energy as you observed that
# constrains the length of the normal of the SDF to 1. This is a regularization
# term to make sure the output is still a valid SDF.

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
    sdf_loss = paddle.sum(paddle.abs(paddle.pow(grid_normal_x[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + paddle.pow(grid_normal_y[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) \
                                   + paddle.pow(grid_normal_z[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1],
                                               2) - 1))   / ((grid_res_x-1) * (grid_res_y-1) * (grid_res_z-1))

    print("\n\nimage loss: ", image_loss.item())
    print("sdf loss: ", sdf_loss.item())

    return image_loss, sdf_loss

#多batchsize版本
def get_fine_pos_and_normal_bs(batchsize, width, height, bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, voxel_size, grid,
                            grid_res_x, grid_res_y, grid_res_z, grid_normal_x, grid_normal_y, grid_normal_z,
                            ray_direction, intersection_pos_rough, voxel_min_point_index, error_code):
    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:, :, :, 0] != error_code).type(Tensor)

    # Get the indices of the minimum point of the intersecting voxels
    x = voxel_min_point_index[:, :, :, 0].type(paddle.cuda.LongTensor)
    y = voxel_min_point_index[:, :, :, 1].type(paddle.cuda.LongTensor)
    z = voxel_min_point_index[:, :, :, 2].type(paddle.cuda.LongTensor)
    x[x == error_code] = 0
    y[y == error_code] = 0
    z[z == error_code] = 0
    x = x.clip(min = 0, max = grid_res_x - 2)
    y = y.clip(min = 0, max = grid_res_y - 2)
    z = z.clip(min = 0, max = grid_res_z - 2)

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    x1 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    x2 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    x3 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    x4 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(3)
    x5 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(3)
    x6 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    x7 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    x8 = paddle.index_select(grid_normal_x.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    intersection_grid_normal_x = paddle.cat((x1, x2, x3, x4, x5, x6, x7, x8), 3) + (
            1 - mask.view(batchsize,height, width, 1).repeat(1,1, 1, 8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    y2 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    y3 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    y4 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(3)
    y5 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
        x.shape).unsqueeze_(3)
    y6 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    y7 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    y8 = paddle.index_select(grid_normal_y.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    intersection_grid_normal_y = paddle.cat((y1, y2, y3, y4, y5, y6, y7, y8), 3) + (
            1 - mask.view(batchsize, height, width, 1).repeat(1, 1, 1, 8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    z2 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    z3 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(
        x.shape).unsqueeze_(3)
    z4 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * x.view(
                                -1)).view(x.shape).unsqueeze_(3)
    z5 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                            x.shape).unsqueeze_(3)
    z6 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    z7 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    z8 = paddle.index_select(grid_normal_z.view(-1), axis = 0,
                            index = (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x * (x + 1).view(
                                -1)).view(x.shape).unsqueeze_(3)
    intersection_grid_normal_z = paddle.cat((z1, z2, z3, z4, z5, z6, z7, z8), 3) + (
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
    intersection_normal = paddle.cat(
        (intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 3)
    intersection_normal = intersection_normal / paddle.unsqueeze(paddle.norm(intersection_normal, p=2, dim=3), 3).repeat(
        1, 1, 1, 3)

    return intersection_pos, intersection_normal

def refraction_bs(l, normal, eta1, eta2):
    # l [b,h,w,3]
    # normal [b,h,w,3]
    # eta1 float
    # eta2 float
    cos_theta = paddle.sum(l * (-normal), dim=3).unsqueeze(3)
    i_p = l + normal * cos_theta
    t_p = eta1 / eta2 * i_p

    t_p_norm = paddle.sum(t_p * t_p, dim=3)
    totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(3)

    t_i = paddle.sqrt(1 - paddle.clip(t_p_norm, min=0, max=0.999999)).unsqueeze(3).expand_as(normal) * (-normal)
    t = t_i + t_p
    t = t / paddle.sqrt(paddle.clip(paddle.sum(t * t, dim=3), min=1e-10)).unsqueeze(3)

    cos_theta_t = paddle.sum(t * (-normal), dim=3).unsqueeze(3)

    e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
          paddle.clip(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
    e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
          paddle.clip(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

    attenuate = paddle.clip(0.5 * (e_i * e_i + e_p * e_p), min=0, max=1)

    return t, attenuate, totalReflectMask

def reflection_bs(l, normal):
    # l n x 3 x imHeight x imWidth
    # normal n x 3 x imHeight x imWidth
    # eta1 float
    # eta2 float
    cos_theta = paddle.sum(l * (-normal), dim=3).unsqueeze(3)
    r_p = l + normal * cos_theta
    r_p_norm = paddle.clip(paddle.sum(r_p * r_p, dim=3), min=0, max=0.999999)
    r_i = paddle.sqrt(1 - r_p_norm).unsqueeze(3).expand_as(normal) * normal
    r = r_p + r_i
    r = r / paddle.sqrt(paddle.clip(paddle.sum(r * r, dim=3), min=1e-10).unsqueeze(3))

    return r

def sampleEnvLight_bs(l, envmap, envHeight, envWidth, batchsize, imHeight, imWidth ):
    offset = np.arange(0, batchsize).reshape([batchsize, 1, 1, 1])
    offset = (offset * envWidth * envHeight).astype(np.int64)
    offset = paddle.to_tensor(offset)

    channelNum = envmap.size(3)

    l = paddle.clip(l, min = -0.999999, max = 0.999999)
    # Compute theta and phi
    x, y, z = paddle.split(l, [1, 1, 1], dim=3)
    theta = paddle.acos(y)
    phi = paddle.atan2(x, z)
    # watch = paddle.sum(z==0)
    v = theta / np.pi * (envHeight - 1)
    u = (-phi / np.pi / 2.0 + 0.5) * (envWidth - 1)

    # Bilinear interpolation to get the new image
    offset = offset.detach()[0:batchsize, :]
    offset = offset.expand_as(u).clone().cuda()

    u, v = paddle.flatten(u), paddle.flatten(v)
    u1 = paddle.clip(paddle.floor(u).detach(), min=0, max=envWidth - 1)
    v1 = paddle.clip(paddle.floor(v).detach(), min=0, max=envHeight - 1)
    u2 = paddle.clip(paddle.ceil(u).detach(), min=0, max=envWidth - 1)
    v2 = paddle.clip(paddle.ceil(v).detach(), min=0, max=envHeight - 1)

    w_r = (u - u1).unsqueeze(1)
    w_l = (1 - w_r)
    w_u = (v2 - v).unsqueeze(1)
    w_d = (1 - w_u)

    u1, v1 = u1.long(), v1.long()
    u2, v2 = u2.long(), v2.long()
    offset = paddle.flatten(offset)
    size_0 = envWidth * envHeight * batchsize
    envmap = envmap.reshape([size_0, channelNum])
    index = (v1 * envWidth + u2) + offset
    envmap_ru = paddle.index_select(envmap, axis = 0, index = index)
    index = (v2 * envWidth + u2) + offset
    envmap_rd = paddle.index_select(envmap, axis = 0, index = index)
    index = (v1 * envWidth + u1) + offset
    envmap_lu = paddle.index_select(envmap, axis = 0, index = index)
    index = (v2 * envWidth + u1) + offset
    envmap_ld = paddle.index_select(envmap, axis = 0, index = index)

    envmap_r = envmap_ru * w_u.expand_as(envmap_ru) + \
               envmap_rd * w_d.expand_as(envmap_rd)
    envmap_l = envmap_lu * w_u.expand_as(envmap_lu) + \
               envmap_ld * w_d.expand_as(envmap_ld)
    renderedImg = envmap_r * w_r.expand_as(envmap_r) + \
                  envmap_l * w_l.expand_as(envmap_l)

    # Post processing
    renderedImg = renderedImg.reshape([batchsize, imHeight, imWidth, channelNum])

    return renderedImg

def transformCoordinate(batchSize, imHeight,imWidth, l, origin, lookat, up ):
    batchSize = origin.size(0 )
    assert(batchSize <= batchSize )

    # Rotate to world coordinate
    zAxis = origin - lookat
    yAxis = up
    xAxis = paddle.cross(yAxis, zAxis, dim=1 )
    xAxis = xAxis / paddle.sqrt(paddle.clip(paddle.sum(xAxis * xAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )
    yAxis = yAxis / paddle.sqrt(paddle.clip(paddle.sum(yAxis * yAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )
    zAxis = zAxis / paddle.sqrt(paddle.clip(paddle.sum(zAxis * zAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )

    xAxis = xAxis.view([batchSize, 3, 1, 1, 1])
    yAxis = yAxis.view([batchSize, 3, 1, 1, 1])
    zAxis = zAxis.view([batchSize, 3, 1, 1, 1])
    rotMat = paddle.cat([xAxis, yAxis, zAxis], dim=2 )
    l = l.unsqueeze(1)

    l = paddle.sum(rotMat.expand([batchSize, 3, 3, imHeight, imWidth ] ) * \
            l.expand([batchSize, 3, 3, imHeight, imWidth ] ), dim=2)
    l = l / paddle.sqrt( paddle.clip(paddle.sum(l*l, dim=1 ).unsqueeze(1), min=1e-10 ) )

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
    v = paddle.to_tensor(v).detach().cuda()

    l = v.repeat([batchSize,1,1,1])
    l = transformCoordinate(batchSize, imHeight,imWidth,l, origin, lookat, up).permute([0,2,3,1])
    backImg = sampleEnvLight_bs(l, envmap, envHeight, envWidth, batchSize, imHeight, imWidth)
    return backImg

