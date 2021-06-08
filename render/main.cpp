#include <iostream>
#include "render.h"
#include <vector>
#include <opencv2/core.hpp>

using namespace std;

void checkCUDAError(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

int main() {
    std::vector<float> a = test_test();
	cout << "test";

	//用立方体或者球 测试 渲染
	int pDim[3]{ 256,256,256 };
	int pImageDim[3]{ 200,200,3 };
	float* watch = new float[256 * 256 * 256];
	float* refraction1 = new float[200 * 200 * 3];
	float* refraction2 = new float[200 * 200 * 3];
	float* intersec_pos2 = new float[200 * 200 * 3];
	float* intersec_nor2 = new float[200 * 200 * 3];

	at::Tensor cube(pDim, 3,1.0);
	cudaMemcpy(watch, cube.data_ptr, sizeof(float)*cube.n, cudaMemcpyDeviceToHost);
	cube.form_cube_sdf(1, 2);
	cudaMemcpy(watch, cube.data_ptr, sizeof(float)*cube.n, cudaMemcpyDeviceToHost);
	

	at::Tensor w_h_3(pImageDim, 3);
	at::Tensor w_h(pImageDim, 2);

	std::vector<at::Tensor> results =  ray_matching(w_h_3, w_h, cube, pImageDim[0], pImageDim[1], -1, -1, -1, 1, 1, 1, 256, 256, 256, 0, 0, -1.5);
	cudaMemcpy(watch, results[0].data_ptr, sizeof(float)*cube.n/256*3, cudaMemcpyDeviceToHost);
	cudaMemcpy(refraction1, results[5].data_ptr, sizeof(float)*200*200*3, cudaMemcpyDeviceToHost);
	cudaMemcpy(refraction2, results[8].data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(intersec_pos2, results[0].data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(intersec_nor2, results[7].data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);
	printf("end");

}
