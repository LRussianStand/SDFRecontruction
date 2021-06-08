#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector <at::Tensor> ray_matching_cuda(
        const at::Tensor w_h_3,
        const at::Tensor w_h,
        const at::Tensor grid,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        const float eye_x,
        const float eye_y,
        const float eye_z
);

std::vector <at::Tensor> ray_matching(
        const at::Tensor w_h_3,
        const at::Tensor w_h,
        const at::Tensor grid,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        const float eye_x,
        const float eye_y,
        const float eye_z) {
    CHECK_INPUT(w_h_3);
    CHECK_INPUT(w_h);
    CHECK_INPUT(grid);

    return ray_matching_cuda(w_h_3, w_h, grid, width, height,
                             bounding_box_min_x,
                             bounding_box_min_y,
                             bounding_box_min_z,
                             bounding_box_max_x,
                             bounding_box_max_y,
                             bounding_box_max_z,
                             grid_res_x,
                             grid_res_y,
                             grid_res_z,
                             eye_x, eye_y, eye_z);
}


//-------------------------------------------------------------------------------------------------------------------

//支持相机自定义
std::vector <at::Tensor> ray_matching_cuda_cam(
        const at::Tensor w_h_3,
        const at::Tensor w_h,
        const at::Tensor grid,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        float fov,
        const at::Tensor eye,
        const at::Tensor lookat,
        const at::Tensor lookup
);

std::vector <at::Tensor> ray_matching_cam(
        const at::Tensor w_h_3,
        const at::Tensor w_h,
        const at::Tensor grid,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        float fov,
        const at::Tensor eye,
        const at::Tensor lookat,
        const at::Tensor lookup) {
    CHECK_INPUT(w_h_3);
    CHECK_INPUT(w_h);
    CHECK_INPUT(grid);

    return ray_matching_cuda_cam(w_h_3, w_h, grid, width, height,
                                 bounding_box_min_x,
                                 bounding_box_min_y,
                                 bounding_box_min_z,
                                 bounding_box_max_x,
                                 bounding_box_max_y,
                                 bounding_box_max_z,
                                 grid_res_x,
                                 grid_res_y,
                                 grid_res_z,
                                 fov,
                                 eye, lookat, lookup);
}

//--------------------------------------------------------------------------------
//一直origins and directions
std::vector <at::Tensor> ray_matching_cuda_dir(
        const at::Tensor w_h_3,
        const at::Tensor w_h,
        const at::Tensor grid,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        const at::Tensor origin,
        const at::Tensor direction,
        const at::Tensor origin_image_distances,
        const at::Tensor pixel_distances
);

std::vector <at::Tensor> ray_matching_dir(
        const at::Tensor w_h_3,
        const at::Tensor w_h,
        const at::Tensor grid,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        const at::Tensor origin,
        const at::Tensor direction,
        const at::Tensor origin_image_distances,
        const at::Tensor pixel_distances) {
    CHECK_INPUT(w_h_3);
    CHECK_INPUT(w_h);
    CHECK_INPUT(grid);

    return ray_matching_cuda_dir(w_h_3, w_h, grid, width, height,
                                 bounding_box_min_x,
                                 bounding_box_min_y,
                                 bounding_box_min_z,
                                 bounding_box_max_x,
                                 bounding_box_max_y,
                                 bounding_box_max_z,
                                 grid_res_x,
                                 grid_res_y,
                                 grid_res_z,
                                 origin,
                                 direction,
                                 origin_image_distances,
                                 pixel_distances);
}

//---------------------------------------------------------------------------------------------------------------------

std::vector<at::Tensor> ray_matching_cuda_cam_bs(
        const at::Tensor b_w_h_3,
        const at::Tensor b_w_h,
        const at::Tensor grid,
        const int batchsize,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        float fov,
        const at::Tensor eye,
        const at::Tensor lookat,
        const at::Tensor lookup,
        const int error_code);

std::vector<at::Tensor> ray_matching_cam_bs(
        const at::Tensor b_w_h_3,
        const at::Tensor b_w_h,
        const at::Tensor grid,
        const int batchsize,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        float fov,
        const at::Tensor eye,
        const at::Tensor lookat,
        const at::Tensor lookup,
        const int error_code)
{
    CHECK_INPUT(b_w_h_3);
    CHECK_INPUT(b_w_h);
    CHECK_INPUT(grid);

    return ray_matching_cuda_cam_bs(b_w_h_3, b_w_h, grid, batchsize, width, height,
                                    bounding_box_min_x,
                                    bounding_box_min_y,
                                    bounding_box_min_z,
                                    bounding_box_max_x,
                                    bounding_box_max_y,
                                    bounding_box_max_z,
                                    grid_res_x,
                                    grid_res_y,
                                    grid_res_z,
                                    fov,
                                    eye, lookat, lookup, error_code);
}

//----------------------------------------------------------------------------------------------------------------------
std::vector <at::Tensor> ray_matching_cuda_dir_bs(
        const at::Tensor b_w_h_3,
        const at::Tensor b_w_h,
        const at::Tensor grid,
        const int batchsize,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        const at::Tensor origin,
        const at::Tensor direction,
        const at::Tensor origin_image_distances,
        const at::Tensor pixel_distances,
        const int error_code
);

std::vector <at::Tensor> ray_matching_dir_bs(
        const at::Tensor b_w_h_3,
        const at::Tensor b_w_h,
        const at::Tensor grid,
        const int batchsize,
        const int width,
        const int height,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        const at::Tensor origin,
        const at::Tensor direction,
        const at::Tensor origin_image_distances,
        const at::Tensor pixel_distances,
        const int error_code) {
    CHECK_INPUT(b_w_h_3);
    CHECK_INPUT(b_w_h);
    CHECK_INPUT(grid);

    return ray_matching_cuda_dir_bs(b_w_h_3, b_w_h, grid, batchsize, width, height,
                                 bounding_box_min_x,
                                 bounding_box_min_y,
                                 bounding_box_min_z,
                                 bounding_box_max_x,
                                 bounding_box_max_y,
                                 bounding_box_max_z,
                                 grid_res_x,
                                 grid_res_y,
                                 grid_res_z,
                                 origin,
                                 direction,
                                 origin_image_distances,
                                 pixel_distances,
                                 error_code);

}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ray_matching", &ray_matching, "Ray Matching");
    m.def("ray_matching_cam", &ray_matching_cam, "Ray Matching with detailed camera parameters");
    m.def("ray_matching_dir", &ray_matching_dir, "Ray Matching with known directions");
    m.def("bs_ray_matching_cam", &ray_matching_cam_bs, "Batch-based Ray Matching with detailed camera parameters");
    m.def("bs_ray_matching_dir", &ray_matching_dir_bs, "Batch-based Ray Matching with known directions");

}
