//
// Created by Administrator on 2020-10-13.
//

#ifndef TRANS_RENDER_RENDER_H
#define TRANS_RENDER_RENDER_H
//
// Created by Administrator on 2020-10-13.
//
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>


namespace at {
	class Tensor
	{
	public:
		Tensor();
		~Tensor();
		Tensor(int* dim, int dim_n, float value = 0);

		//复制构造函数
		Tensor(const Tensor & src);
		template<typename T> T* data() const;

		//转化成立方体sdf，grid的初始化在构造函数里面,w为立方体的宽度,bw为grid（boundingbox）宽度
		void form_cube_sdf(float w,float bw);

	private:

	public:

		float* data_ptr;
		int n;

		int* dim;
		int dim_n;

	};
	Tensor zeros_like(const Tensor & a);
}

//---------------------------------------------------------------------------------------------------

std::vector<at::Tensor> ray_matching_cuda(
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
        const float eye_z);

std::vector<float> test_test();

//#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> ray_matching(
	const at::Tensor& w_h_3,
	const at::Tensor& w_h,
	const at::Tensor& grid,
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
	const float eye_z);



#endif //TRANS_RENDER_RENDER_H
