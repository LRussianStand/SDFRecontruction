#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <ATen/ATen.h>

#define PI 3.14159265358979323846f
#define FLT_MAX -1
#define FLT_EPSILON      1.192092896e-07F

//变换角度单位 from degree to radiance
__device__ __forceinline__ float DegToRad(const float &deg) { return (deg * (PI / 180.f)); }

//计算三维向量长度（模）
__device__ __forceinline__ float length(
	const float x,
	const float y,
	const float z) {
	return sqrtf(x*x + y*y + z*z);
}

// Cross product x
__device__ __forceinline__ float cross_x(
	const float a_x,
	const float a_y,
	const float a_z,
	const float b_x,
	const float b_y,
	const float b_z) {
	return a_y * b_z - a_z * b_y;
}

// Cross product y
__device__ __forceinline__ float cross_y(
	const float a_x,
	const float a_y,
	const float a_z,
	const float b_x,
	const float b_y,
	const float b_z) {
	return a_z * b_x - a_x * b_z;
}

// Cross product z
__device__ __forceinline__ float cross_z(
	const float a_x,
	const float a_y,
	const float a_z,
	const float b_x,
	const float b_y,
	const float b_z) {
	return a_x * b_y - a_y * b_x;
}

//origins[w*h*3]每个像素相机中心，directions[w*h*3]每个像素方向，origin_image_distances[w*h]每个像素到相机距离，pixel_distances[w*h]像素宽度（总宽为1）
__global__ void GenerateRay2(
	float* origins,
	float* directions,
	float* origin_image_distances,
	float* pixel_distances,
	const int width,
	const int height,
	const float fov,
	const float* eye, const float* lookat, const float* lookup) {

	const float eye_x = eye[0];
	const float eye_y = eye[1];
	const float eye_z = eye[2];
	const float at_x = lookat[0];
	const float at_y = lookat[1];
	const float at_z = lookat[2];
	const float up_x = lookup[0];
	const float up_y = lookup[1];
	const float up_z = lookup[2];

	// Compute camera view volume（默认 focal为1,视口上角度为30° 的情况下，上下左右边界）
//	const float top = tan(DegToRad(fov));
//	const float bottom = -top;
//	const float right = (__int2float_rd(width) / __int2float_rd(height)) * top;
//	const float left = -right;

	const float right = tan(DegToRad(fov));
	const float left = -right;
    const float top = (__int2float_rd(height) / __int2float_rd(width)) * right;
    const float bottom = -top;

	// Compute local base （eye为相机原点坐标，这一步计算相机坐标空间xyz轴的方向向量）
	const float w_x = -(eye_x - at_x) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
	const float w_y = -(eye_y - at_y) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
	const float w_z = -(eye_z - at_z) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
	const float cross_up_w_x = cross_x(up_x, up_y, up_z, w_x, w_y, w_z);
	const float cross_up_w_y = cross_y(up_x, up_y, up_z, w_x, w_y, w_z);
	const float cross_up_w_z = cross_z(up_x, up_y, up_z, w_x, w_y, w_z);
	const float u_x = (cross_up_w_x) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
	const float u_y = (cross_up_w_y) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
	const float u_z = (cross_up_w_z) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
	const float v_x = cross_x(w_x, w_y, w_z, u_x, u_y, u_z);
	const float v_y = cross_y(w_x, w_y, w_z, u_x, u_y, u_z);
	const float v_z = cross_z(w_x, w_y, w_z, u_x, u_y, u_z);

	// 像素运算 y 为 0？
	const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (pixel_index < width * height) {
		const int x = pixel_index % width;
		const int y = pixel_index / width;
		const int i = 3 * pixel_index;

		// Compute point on view plane
		// Ray passes through the center of the pixel
		const float view_plane_x = left + (right - left) * (__int2float_rd(x) + 0.5) / __int2float_rd(width);
		const float view_plane_y = top - (top - bottom) * (__int2float_rd(y) + 0.5) / __int2float_rd(height);
		const float s_x = view_plane_x * u_x + view_plane_y * v_x + w_x;
		const float s_y = view_plane_x * u_y + view_plane_y * v_y + w_y;
		const float s_z = view_plane_x * u_z + view_plane_y * v_z + w_z;
		origins[i] = eye_x;
		origins[i + 1] = eye_y;
		origins[i + 2] = eye_z;


		directions[i] = s_x / length(s_x, s_y, s_z);
		directions[i + 1] = s_y / length(s_x, s_y, s_z);
		directions[i + 2] = s_z / length(s_x, s_y, s_z);

		origin_image_distances[pixel_index] = length(s_x, s_y, s_z);
		pixel_distances[pixel_index] = (right - left) / __int2float_rd(width);

	}
}

__global__ void GenerateRay(
	float* origins,
	float* directions,
	float* origin_image_distances,
	float* pixel_distances,
	const int width,
	const int height,
	const float eye_x,
	const float eye_y,
	const float eye_z) {

	const float at_x = 0;
	const float at_y = 0;
	const float at_z = 0;
	const float up_x = 0;
	const float up_y = 1;
	const float up_z = 0;

	// Compute camera view volume（默认 focal为1,视口上角度为30° 的情况下，上下左右边界）
	const float top = tan(DegToRad(30));
	const float bottom = -top;
	const float right = (__int2float_rd(width) / __int2float_rd(height)) * top;
	const float left = -right;

	// Compute local base （eye为相机原点坐标，这一步计算相机坐标空间xyz轴的方向向量）
	const float w_x = -(eye_x - at_x) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
	const float w_y = -(eye_y - at_y) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
	const float w_z = -(eye_z - at_z) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
	const float cross_up_w_x = cross_x(up_x, up_y, up_z, w_x, w_y, w_z);
	const float cross_up_w_y = cross_y(up_x, up_y, up_z, w_x, w_y, w_z);
	const float cross_up_w_z = cross_z(up_x, up_y, up_z, w_x, w_y, w_z);
	const float u_x = (cross_up_w_x) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
	const float u_y = (cross_up_w_y) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
	const float u_z = (cross_up_w_z) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
	const float v_x = cross_x(w_x, w_y, w_z, u_x, u_y, u_z);
	const float v_y = cross_y(w_x, w_y, w_z, u_x, u_y, u_z);
	const float v_z = cross_z(w_x, w_y, w_z, u_x, u_y, u_z);

	// 像素运算 y 为 0？
	const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (pixel_index < width * height) {
		const int x = pixel_index % width;
		const int y = pixel_index / width;
		const int i = 3 * pixel_index;

		// Compute point on view plane
		// Ray passes through the center of the pixel
		const float view_plane_x = left + (right - left) * (__int2float_rd(x) + 0.5) / __int2float_rd(width);
		const float view_plane_y = top - (top - bottom) * (__int2float_rd(y) + 0.5) / __int2float_rd(height);
		const float s_x = view_plane_x * u_x + view_plane_y * v_x + w_x;
		const float s_y = view_plane_x * u_y + view_plane_y * v_y + w_y;
		const float s_z = view_plane_x * u_z + view_plane_y * v_z + w_z;
		origins[i] = eye_x;
		origins[i + 1] = eye_y;
		origins[i + 2] = eye_z;


		directions[i] = s_x / length(s_x, s_y, s_z);
		directions[i + 1] = s_y / length(s_x, s_y, s_z);
		directions[i + 2] = s_z / length(s_x, s_y, s_z);

		origin_image_distances[pixel_index] = length(s_x, s_y, s_z);
		pixel_distances[pixel_index] = (right - left) / __int2float_rd(width);

	}
}

// Check if a point is inside
__device__ __forceinline__ bool InsideBoundingBox(
	const float p_x,
	const float p_y,
	const float p_z,
	const float bounding_box_min_x,
	const float bounding_box_min_y,
	const float bounding_box_min_z,
	const float bounding_box_max_x,
	const float bounding_box_max_y,
	const float bounding_box_max_z) {

	return (p_x >= bounding_box_min_x) && (p_x <= bounding_box_max_x) &&
		(p_y >= bounding_box_min_y) && (p_y <= bounding_box_max_y) &&
		(p_z >= bounding_box_min_z) && (p_z <= bounding_box_max_z);
}

// Compute the distance along the ray between the point and the bounding box
// 如果射线能够到bounding box，则距离计算准确；如果不能到达，则 1. 已经在外，无望碰触返回-1 2.部分在内，部分在外，返回fake 距离
__device__ float Distance(
	const float reached_point_x,
	const float reached_point_y,
	const float reached_point_z,
	float direction_x,
	float direction_y,
	float direction_z,
	const float bounding_box_min_x,
	const float bounding_box_min_y,
	const float bounding_box_min_z,
	const float bounding_box_max_x,
	const float bounding_box_max_y,
	const float bounding_box_max_z) {

	float dist = -1.f;
	direction_x = direction_x / length(direction_x, direction_y, direction_z);
	direction_y = direction_y / length(direction_x, direction_y, direction_z);
	direction_z = direction_z / length(direction_x, direction_y, direction_z);

	// For each axis count any excess distance outside box extents
	float v = reached_point_x;
	float d = direction_x;
	if (dist == -1) {
		if ((v < bounding_box_min_x) && (d > 0)) { dist = (bounding_box_min_x - v) / d; }
		if ((v > bounding_box_max_x) && (d < 0)) { dist = (bounding_box_max_x - v) / d; }
	}
	else {
		if ((v < bounding_box_min_x) && (d > 0)) { dist = fmaxf(dist, (bounding_box_min_x - v) / d); }
		if ((v > bounding_box_max_x) && (d < 0)) { dist = fmaxf(dist, (bounding_box_max_x - v) / d); }
	}

	v = reached_point_y;
	d = direction_y;
	if (dist == -1) {
		if ((v < bounding_box_min_y) && (d > 0)) { dist = (bounding_box_min_y - v) / d; }
		if ((v > bounding_box_max_y) && (d < 0)) { dist = (bounding_box_max_y - v) / d; }
	}
	else {
		if ((v < bounding_box_min_y) && (d > 0)) { dist = fmaxf(dist, (bounding_box_min_y - v) / d); }
		if ((v > bounding_box_max_y) && (d < 0)) { dist = fmaxf(dist, (bounding_box_max_y - v) / d); }
	}

	v = reached_point_z;
	d = direction_z;
	if (dist == -1) {
		if ((v < bounding_box_min_z) && (d > 0)) { dist = (bounding_box_min_z - v) / d; }
		if ((v > bounding_box_max_z) && (d < 0)) { dist = (bounding_box_max_z - v) / d; }
	}
	else {
		if ((v < bounding_box_min_z) && (d > 0)) { dist = fmaxf(dist, (bounding_box_min_z - v) / d); }
		if ((v > bounding_box_max_z) && (d < 0)) { dist = fmaxf(dist, (bounding_box_max_z - v) / d); }
	}

	return dist;
}

//三维坐标转1维，主x - y - z
__device__ __forceinline__ int flat(float const x, float const y, float const z,
	int const grid_res_x, int const grid_res_y, int const grid_res_z) {
	return __int2float_rd(z) + __int2float_rd(y) * grid_res_z + __int2float_rd(x) * grid_res_z * grid_res_y;
}

// Get the signed distance value at the specific point
// 没到bounding box，则计算到bounding box距离，到了 就计算插值sdf距离
__device__ float ValueAt(
	const float* grid,
	const float reached_point_x,
	const float reached_point_y,
	const float reached_point_z,
	const float direction_x,
	const float direction_y,
	const float direction_z,
	const float bounding_box_min_x,
	const float bounding_box_min_y,
	const float bounding_box_min_z,
	const float bounding_box_max_x,
	const float bounding_box_max_y,
	const float bounding_box_max_z,
	const int grid_res_x,
	const int grid_res_y,
	const int grid_res_z,
	const bool first_time) {

	// Check if we are outside the BBOX
	if (!InsideBoundingBox(reached_point_x, reached_point_y, reached_point_z,
		bounding_box_min_x,
		bounding_box_min_y,
		bounding_box_min_z,
		bounding_box_max_x,
		bounding_box_max_y,
		bounding_box_max_z)) {

		// If it is the first time, then the ray has not entered the grid
		if (first_time) {

			return Distance(reached_point_x, reached_point_y, reached_point_z,
				direction_x, direction_y, direction_z,
				bounding_box_min_x,
				bounding_box_min_y,
				bounding_box_min_z,
				bounding_box_max_x,
				bounding_box_max_y,
				bounding_box_max_z) + 0.00001f;
		}

		// Otherwise, the ray has left the grid
		else {
			return -1;
		}
	}

	// Compute voxel size
	float voxel_size = (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1);

	// Compute the the minimum point of the intersecting voxel
	float min_index_x = floorf((reached_point_x - bounding_box_min_x) / voxel_size);
	float min_index_y = floorf((reached_point_y - bounding_box_min_y) / voxel_size);
	float min_index_z = floorf((reached_point_z - bounding_box_min_z) / voxel_size);

	// Check whether the ray intersects the vertex with the last index of the axis
	// If so, we should record the previous index （这个部分可以直接替换为 使用max_index_x来计算！！！）
	if (min_index_x == (bounding_box_max_x - bounding_box_min_x) / voxel_size) {
		min_index_x = (bounding_box_max_x - bounding_box_min_x) / voxel_size - 1;
	}
	if (min_index_y == (bounding_box_max_y - bounding_box_min_y) / voxel_size) {
		min_index_y = (bounding_box_max_y - bounding_box_min_y) / voxel_size - 1;
	}
	if (min_index_z == (bounding_box_max_z - bounding_box_min_z) / voxel_size) {
		min_index_z = (bounding_box_max_z - bounding_box_min_z) / voxel_size - 1;
	}

	// Linear interpolate along x axis the eight values
	const float tx = (reached_point_x - (bounding_box_min_x + min_index_x * voxel_size)) / voxel_size;
	const float c01 = (1.f - tx) * grid[flat(min_index_x, min_index_y, min_index_z, grid_res_x, grid_res_y, grid_res_z)]
		+ tx * grid[flat(min_index_x + 1, min_index_y, min_index_z, grid_res_x, grid_res_y, grid_res_z)];
	const float c23 = (1.f - tx) * grid[flat(min_index_x, min_index_y + 1, min_index_z, grid_res_x, grid_res_y, grid_res_z)]
		+ tx * grid[flat(min_index_x + 1, min_index_y + 1, min_index_z, grid_res_x, grid_res_y, grid_res_z)];
	const float c45 = (1.f - tx) * grid[flat(min_index_x, min_index_y, min_index_z + 1, grid_res_x, grid_res_y, grid_res_z)]
		+ tx * grid[flat(min_index_x + 1, min_index_y, min_index_z + 1, grid_res_x, grid_res_y, grid_res_z)];
	const float c67 = (1.f - tx) * grid[flat(min_index_x, min_index_y + 1, min_index_z + 1, grid_res_x, grid_res_y, grid_res_z)]
		+ tx * grid[flat(min_index_x + 1, min_index_y + 1, min_index_z + 1, grid_res_x, grid_res_y, grid_res_z)];

	// Linear interpolate along the y axis
	const float ty = (reached_point_y - (bounding_box_min_y + min_index_y * voxel_size)) / voxel_size;
	const float c0 = (1.f - ty) * c01 + ty * c23;
	const float c1 = (1.f - ty) * c45 + ty * c67;

	// Return final value interpolated along z
	const float tz = (reached_point_z - (bounding_box_min_z + min_index_z * voxel_size)) / voxel_size;

	return (1.f - tz) * c0 + tz * c1;
}

// Compute the intersection of the ray and the grid
// The intersection procedure uses ray marching to check if we have an interaction with the stored surface
// 添加了 让射线反向的inv, 正常inv为1，若反向则设置为-1
__global__ void Intersect(
	const float* grid,
	const float* origins,
	const float* directions,
	const float* origin_image_distances,
	const float* pixel_distances,
	const float bounding_box_min_x,
	const float bounding_box_min_y,
	const float bounding_box_min_z,
	const float bounding_box_max_x,
	const float bounding_box_max_y,
	const float bounding_box_max_z,
	const int grid_res_x,
	const int grid_res_y,
	const int grid_res_z,
	float* voxel_position,
	float* intersection_pos,
	const int width,
	const int height,
	const int inv) {

	// Compute voxel size
	const float voxel_size = (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1);

	// Define constant values
	const int max_steps = 1000;
	bool first_time = true;
	float depth = 0;
	int gotten_result = 0;

	const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (pixel_index < width * height) {

		const int i = 3 * pixel_index;

		if (origins[i] == FLT_MAX)
		{
			intersection_pos[i] = FLT_MAX;
			intersection_pos[i + 1] = FLT_MAX;
			intersection_pos[i + 2] = FLT_MAX;
			voxel_position[i] = FLT_MAX;
			voxel_position[i + 1] = FLT_MAX;
			voxel_position[i + 2] = FLT_MAX;
			return;
		}

		for (int steps = 0; steps < max_steps; steps++) {

			float reached_point_x = origins[i] + inv * depth * directions[i];
			float reached_point_y = origins[i + 1] + inv * depth * directions[i + 1];
			float reached_point_z = origins[i + 2] + inv * depth * directions[i + 2];

			// Get the signed distance value for the point the ray reaches
			const float distance = ValueAt(grid, reached_point_x, reached_point_y, reached_point_z,
				inv * directions[i], inv * directions[i + 1], inv * directions[i + 2],
				bounding_box_min_x,
				bounding_box_min_y,
				bounding_box_min_z,
				bounding_box_max_x,
				bounding_box_max_y,
				bounding_box_max_z,
				grid_res_x,
				grid_res_y,
				grid_res_z, first_time);
			first_time = false;

			// Check if the ray is going ourside the bounding box
			if (distance == -1) {
				voxel_position[i] = FLT_MAX;
				voxel_position[i + 1] = FLT_MAX;
				voxel_position[i + 2] = FLT_MAX;
				intersection_pos[i] = FLT_MAX;
				intersection_pos[i + 1] = FLT_MAX;
				intersection_pos[i + 2] = FLT_MAX;
				gotten_result = 1;
				break;
			}

			// Check if we are close enough to the surface
			if (distance < pixel_distances[pixel_index] / origin_image_distances[pixel_index] * depth / 2) {
				//if (distance < 0.1) {

				// Compute the the minimum point of the intersecting voxel
				voxel_position[i] = floorf((reached_point_x - bounding_box_min_x) / voxel_size);
				voxel_position[i + 1] = floorf((reached_point_y - bounding_box_min_y) / voxel_size);
				voxel_position[i + 2] = floorf((reached_point_z - bounding_box_min_z) / voxel_size);
				if (voxel_position[i] == grid_res_x - 1) {
					voxel_position[i] = voxel_position[i] - 1;
				}
				if (voxel_position[i + 1] == grid_res_y - 1) {
					voxel_position[i + 1] = voxel_position[i + 1] - 1;
				}
				if (voxel_position[i + 2] == grid_res_z - 1) {
					voxel_position[i + 2] = voxel_position[i + 2] - 1;
				}
				intersection_pos[i] = reached_point_x;
				intersection_pos[i + 1] = reached_point_y;
				intersection_pos[i + 2] = reached_point_z;
				gotten_result = 1;
				break;
			}

			// Increase distance
			depth += distance;

		}

		if (gotten_result == 0) {

			// No intersections
			voxel_position[i] = FLT_MAX;
			voxel_position[i + 1] = FLT_MAX;
			voxel_position[i + 2] = FLT_MAX;
			intersection_pos[i] = FLT_MAX;
			intersection_pos[i + 1] = FLT_MAX;
			intersection_pos[i + 2] = FLT_MAX;
		}
	}
}

//-----------------------------------------------------------------------------------------

//基于voxel的坐标(float!)，获取某个voxel的法向量
__device__ void get_vox_normal(const float* grid, const float vox_size, const int grid_res_x, const int grid_res_y, const int grid_res_z, float vx, float vy, float vz, float *nor)
{

	//计算前后左右上下的idex
	int pos = vx * grid_res_y * grid_res_z + vy * grid_res_z + vz;
	int posxmin = (vx - 1) * grid_res_y * grid_res_z + vy * grid_res_z + vz;
	int posxmax = (vx + 1) * grid_res_y * grid_res_z + vy * grid_res_z + vz;
	int posymin = vx * grid_res_y * grid_res_z + (vy - 1) * grid_res_z + vz;
	int posymax = vx * grid_res_y * grid_res_z + (vy + 1) * grid_res_z + vz;
	int poszmin = vx * grid_res_y * grid_res_z + vy * grid_res_z + vz - 1;
	int poszmax = vx * grid_res_y * grid_res_z + vy * grid_res_z + vz + 1;

	//计算nx,ny,nz
	float nx = 0;
	float ny = 0;
	float nz = 0;

	if (vx - 1 >= 0 || vx + 1 < grid_res_x - 1)
	{
		nx = (grid[posxmax] - grid[posxmin]) / (2 * vox_size);
	}
	else if (vx < 1)
	{
		float sdf_min = 3 * (grid[pos] - grid[posxmax]) + grid[int((vx + 2) * grid_res_y * grid_res_z + vy * grid_res_z + vz)];
		nx = (grid[posxmax] - sdf_min) / (2 * vox_size);
	}
	else
	{
		float sdf_max = 3 * (grid[posxmax] - grid[pos]) + grid[int((vx - 2) * grid_res_y * grid_res_z + vy * grid_res_z + vz)];
		nx = (sdf_max - grid[posxmin]) / (2 * vox_size);
	}

	if (vy - 1 >= 0 || vy + 1 < grid_res_y - 1)
	{
		ny = (grid[posymax] - grid[posymin]) / (2 * vox_size);
	}
	else if (vy < 1)
	{
		float sdf_min = 3 * (grid[pos] - grid[posymax]) + grid[int((vy + 2) * grid_res_y * grid_res_z + vy * grid_res_z + vz)];
		ny = (grid[posymax] - sdf_min) / (2 * vox_size);
	}
	else
	{
		float sdf_max = 3 * (grid[posymax] - grid[pos]) + grid[int((vy - 2) * grid_res_y * grid_res_z + vy * grid_res_z + vz)];
		ny = (sdf_max - grid[posymin]) / (2 * vox_size);
	}

	if (vz - 1 >= 0 || vz + 1 < grid_res_z - 1)
	{
		nz = (grid[poszmax] - grid[poszmin]) / (2 * vox_size);
	}
	else if (vz < 1)
	{
		float sdf_min = 3 * (grid[pos] - grid[poszmax]) + grid[int((vz + 2) * grid_res_y * grid_res_z + vy * grid_res_z + vz)];
		nz = (grid[poszmax] - sdf_min) / (2 * vox_size);
	}
	else
	{
		float sdf_max = 3 * (grid[poszmax] - grid[pos]) + grid[int((vz - 2) * grid_res_y * grid_res_z + vy * grid_res_z + vz)];
		nz = (sdf_max - grid[poszmin]) / (2 * vox_size);
	}

	float dis = sqrtf(nx*nx + ny * ny + nz * nz);
	nor[0] = nx/dis;
	nor[1] = ny/dis;
	nor[2] = nz/dis;
}


//基于交点、交点voxel坐标、sdf的grid，求交点位置对应的法向量。intersect_normal [w,h,3]
__global__ void IntersectNormal(const float* grid, const float* intersect_pos, const float* voxel_pos, int width, int height, float vox_size, float grid_res_x, float grid_res_y, float grid_res_z, float bx_min, float by_min, float bz_min,float* intersect_normal, int inv)
{
	const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;


	if (pixel_index < width * height) {

		const int i = 3 * pixel_index;

		if (intersect_pos[i] == FLT_MAX)
		{
			intersect_normal[i] = FLT_MAX;
			intersect_normal[i + 1] = FLT_MAX;
			intersect_normal[i + 2] = FLT_MAX;
			return;
		}

		float x = intersect_pos[i];
		float y = intersect_pos[i + 1];
		float z = intersect_pos[i + 2];
		float vx = voxel_pos[i];
		float vy = voxel_pos[i + 1];
		float vz = voxel_pos[i + 2];

		//计算领域8个点的法向量
		float vox_nor[8][3];
		float dvx[8] = { 0,1,0,1,0,1,0,1 };
		float dvy[8] = { 0,0,1,1,0,0,1,1 };
		float dvz[8] = { 0,0,0,0,1,1,1,1 };

		for (int k = 0; k < 8; ++k)
		{
			get_vox_normal(grid, vox_size, grid_res_x, grid_res_y, grid_res_z, vx + dvx[k], vy + dvy[k], vz + dvz[k], vox_nor[k]);
		}

		//获取一个位置的法向量
		float normal[3];
		const float tx = (x - (vx * vox_size + bx_min)) / vox_size;
		const float ty = (y - (vy * vox_size + by_min)) / vox_size;
		const float tz = (z - (vz * vox_size + bz_min)) / vox_size;

		for (int k = 0; k < 3; ++k)
		{
			const float c01 = (1 - tx) * vox_nor[0][k] + tx * vox_nor[1][k];
			const float c23 = (1 - tx) * vox_nor[2][k] + tx * vox_nor[3][k];
			const float c45 = (1 - tx) * vox_nor[4][k] + tx * vox_nor[5][k];
			const float c67 = (1 - tx) * vox_nor[6][k] + tx * vox_nor[7][k];

			const float c0 = (1.f - ty) * c01 + ty * c23;
			const float c1 = (1.f - ty) * c45 + ty * c67;

			normal[k] = (1.f - tz) * c0 + tz * c1;
		}

		float dis = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
		intersect_normal[i] = inv * normal[0] / dis;
		intersect_normal[i + 1] = inv * normal[1] / dis;
		intersect_normal[i + 2] = inv * normal[2] / dis;
	}


}

//根据入射光、法向量 以及 折射率，计算出射光. eta为折射系数
__global__ void Refraction(const float* intersect_pos, const float* ray_dir,const float* nor_dir, int width, int height, float eta1, float eta2,float* refract_dir, float* attenuation, float* total_reflect_mask)
{
	const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;


	if (pixel_index < width * height) {

		const int i = 3 * pixel_index;

		//如果没有碰撞，那么折射光还是ray direction, attenuation赋值为0
		if (intersect_pos[i] == FLT_MAX)
		{
			refract_dir[i] = ray_dir[i];
			refract_dir[i + 1] = ray_dir[i + 1];
			refract_dir[i + 2] = ray_dir[i + 2];
			attenuation[i / 3] = 0.0;
			return;
		}

		//发生碰撞，计算折射角
		float ray_dir_x = ray_dir[i];
		float ray_dir_y = ray_dir[i + 1];
		float ray_dir_z = ray_dir[i + 2];
		float nor_x = nor_dir[i];
		float nor_y = nor_dir[i + 1];
		float nor_z = nor_dir[i + 2];

		float cos1 = -(ray_dir_x * nor_x + ray_dir_y * nor_y + ray_dir_z * nor_z);
		float ray_p_x = ray_dir_x + nor_x * cos1;
		float ray_p_y = ray_dir_y + nor_y * cos1;
		float ray_p_z = ray_dir_z + nor_z * cos1;
		float refrac_p_x = (ray_p_x) * eta1 / eta2;
		float refrac_p_y = (ray_p_y) * eta1 / eta2;
		float refrac_p_z = (ray_p_z) * eta1 / eta2;

		//发生完全反射，折射角全部为max？
		if (length(refrac_p_x, refrac_p_y, refrac_p_z) >= 1)
		{
			total_reflect_mask[i / 3] = 1;
			refract_dir[i] = FLT_MAX;
			refract_dir[i + 1] = FLT_MAX;
			refract_dir[i + 2] = FLT_MAX;
			return;
		}

		float len_refrac_nor = sqrtf(1 - refrac_p_x * refrac_p_x + refrac_p_y * refrac_p_y + refrac_p_z * refrac_p_z);
		refract_dir[i] = -len_refrac_nor * nor_x + refrac_p_x;
		refract_dir[i + 1] = -len_refrac_nor * nor_y + refrac_p_y;
		refract_dir[i + 2] = -len_refrac_nor * nor_z + refrac_p_z;
		float len_refrac = sqrtf( refract_dir[i] * refract_dir[i] + refract_dir[i + 1] * refract_dir[i + 1] + refract_dir[i + 2] * refract_dir[i + 2]);
		refract_dir[i] /= len_refrac;
		refract_dir[i + 1] /= len_refrac;
		refract_dir[i + 2] /= len_refrac;

		float cos2 = -(refract_dir[i] * nor_x + refract_dir[i + 1] * nor_y + refract_dir[i + 2] * nor_z);
		float atten1 = (cos1*eta1 - cos2 * eta2) / (cos1*eta1 + cos2 * eta2 + FLT_EPSILON);
		float atten2 = (cos1*eta2 - cos2 * eta1) / (cos1*eta2 + cos2 * eta1 + FLT_EPSILON);
		attenuation[i / 3] = 0.5 * atten1 * atten1 + 0.5 * atten2 * atten2;

	}

}

//根据入射光、法向量, 计算出反射光
__global__ void Reflection(const float* intersect_pos, const float* ray_dir, const float* nor_dir, int width, int height, float eta1, float eta2, float* reflection)
{
	const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;


	if (pixel_index < width * height) {

		const int i = 3 * pixel_index;

		//如果没有碰撞，那么折射光还是ray direction, attenuation赋值为0
		if (intersect_pos[i] == FLT_MAX)
		{
			reflection[i] = ray_dir[i];
			reflection[i + 1] = ray_dir[i + 1];
			reflection[i + 2] = ray_dir[i + 2];
			return;
		}

		//发生碰撞，计算折射角
		float ray_dir_x = ray_dir[i];
		float ray_dir_y = ray_dir[i + 1];
		float ray_dir_z = ray_dir[i + 2];
		float nor_x = nor_dir[i];
		float nor_y = nor_dir[i + 1];
		float nor_z = nor_dir[i + 2];

		float cos1 = -(ray_dir_x * nor_x + ray_dir_y * nor_y + ray_dir_z * nor_z);
		float ray_p_x = ray_dir_x + nor_x * cos1;
		float ray_p_y = ray_dir_y + nor_y * cos1;
		float ray_p_z = ray_dir_z + nor_z * cos1;

		reflection[i] = 2 * ray_p_x - ray_dir_x;
		reflection[i + 1] = 2 * ray_p_y - ray_dir_y;
		reflection[i + 2] = 2 * ray_p_z - ray_dir_z;

		float len_refrac = sqrtf(reflection[i] * reflection[i] + reflection[i + 1] * reflection[i + 1] + reflection[i + 2] * reflection[i + 2]);
		reflection[i] /= len_refrac;
		reflection[i + 1] /= len_refrac;
		reflection[i + 2] /= len_refrac;


	}

}

//根据 入碰撞位置、入折射方向，找到 反向的起点和方向（不用计算，就是折射光线的反向）。
__global__ void GetInvOrigin(const float* intersect_pos, const float* refraction, int width, int height, float bminx, float bminy, float bminz, float bmaxx, float bmaxy, float bmaxz, float* inv_origin)
{
	const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;


	if (pixel_index < width * height) {

		const int i = 3 * pixel_index;

		//如果没有碰撞，那么反向的位置和方向都是0
		if (intersect_pos[i] == FLT_MAX)
		{
			inv_origin[i] = FLT_MAX;
			inv_origin[i + 1] = FLT_MAX;
			inv_origin[i + 2] = FLT_MAX;
			return;
		}

		//发生碰撞，计算射线与boundingbox的交点
		const float refra_dir_x = refraction[i];
		const float refra_dir_y = refraction[i + 1];
		const float refra_dir_z = refraction[i + 2];
		const float pos_x = intersect_pos[i];
		const float pos_y = intersect_pos[i + 1];
		const float pos_z = intersect_pos[i + 2];

		float fmindis = 1000000000;
		if (refra_dir_x > 0)
		{
			float dis = (bmaxx - pos_x) / refra_dir_x;
			fmindis = fmindis < dis ? fmindis : dis;
		}
		else if (refra_dir_x < 0)
		{
			float dis = -(pos_x - bminx) / refra_dir_x;
			fmindis = fmindis < dis ? fmindis : dis;
		}

		if (refra_dir_y > 0)
		{
			float dis = (bmaxy - pos_y) / refra_dir_y;
			fmindis = fmindis < dis ? fmindis : dis;
		}
		else if (refra_dir_y < 0)
		{
			float dis = -(pos_y - bminy) / refra_dir_y;
			fmindis = fmindis < dis ? fmindis : dis;
		}

		if (refra_dir_z > 0)
		{
			float dis = (bmaxz - pos_z) / refra_dir_z;
			fmindis = fmindis < dis ? fmindis : dis;
		}
		else if (refra_dir_z < 0)
		{
			float dis = -(pos_z - bminz) / refra_dir_z;
			fmindis = fmindis < dis ? fmindis : dis;
		}
		fmindis += FLT_EPSILON * 10;

		inv_origin[i] = pos_x + fmindis * refra_dir_x;
		inv_origin[i + 1] = pos_y +fmindis * refra_dir_y;
		inv_origin[i + 2] = pos_z +fmindis * refra_dir_z;

	}

}



// Ray marching to get the first corner position of the voxel the ray intersects
// 如何计算normal?
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
	const float eye_z) {

//	float* watch = new float[200 * 200];
//	float* watch3 = new float[200 * 200 * 3];
//	float* watch31 = new float[120000];
//	float* watch32 = new float[120000];

	const int thread = 512;

	at::Tensor origins = at::zeros_like(w_h_3);
	at::Tensor directions = at::zeros_like(w_h_3);
	at::Tensor origin_image_distances = at::zeros_like(w_h);
	at::Tensor pixel_distances = at::zeros_like(w_h);


	GenerateRay << <(width * height + thread - 1) / thread, thread >> > (
		origins.data<float>(),
		directions.data<float>(),
		origin_image_distances.data<float>(),
		pixel_distances.data<float>(),
		width,
		height,
		eye_x,
		eye_y,
		eye_z);

//	cudaMemcpy(watch31, directions.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);

	at::Tensor voxel_position = at::zeros_like(w_h_3);
	at::Tensor intersection_pos = at::zeros_like(w_h_3);
	Intersect << <(width * height + thread - 1) / thread, thread >> > (
		grid.data<float>(),
		origins.data<float>(),
		directions.data<float>(),
		origin_image_distances.data<float>(),
		pixel_distances.data<float>(),
		bounding_box_min_x,
		bounding_box_min_y,
		bounding_box_min_z,
		bounding_box_max_x,
		bounding_box_max_y,
		bounding_box_max_z,
		grid_res_x,
		grid_res_y,
		grid_res_z,
		voxel_position.data<float>(),
		intersection_pos.data<float>(),
		width,
		height,
		1);

	at::Tensor intersect_nor = at::zeros_like(w_h_3);
	IntersectNormal << <(width * height + thread - 1) / thread, thread >> > (
		grid.data<float>(),
		intersection_pos.data<float>(),
		voxel_position.data<float>(),
		width, height,
		(bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1),
		grid_res_x, grid_res_y, grid_res_y,
		bounding_box_min_x,bounding_box_min_y,bounding_box_min_z,
		intersect_nor.data<float>(),
		1
		);

	at::Tensor refraction_dir1 = at::zeros_like(w_h_3);
	at::Tensor attenuation1 = at::zeros_like(w_h);
	at::Tensor total_reflect_mask = at::zeros_like(w_h);
	Refraction << <(width * height + thread - 1) / thread, thread >> > (
		intersection_pos.data<float>(),
		directions.data<float>(),
		intersect_nor.data<float>(),
		width, height,
		1, 1.4723,
		refraction_dir1.data<float>(),
		attenuation1.data<float>(),
		total_reflect_mask.data<float>()
		);

//	cudaMemcpy(watch3, refraction_dir1.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);
//	cudaMemcpy(watch, attenuation1.data_ptr, sizeof(float) * 200 * 200 * 1, cudaMemcpyDeviceToHost);
//	//cudaMemcpy(watch, total_reflect_mask.data_ptr, sizeof(float) * 200 * 200 * 1, cudaMemcpyDeviceToHost);

    at::Tensor reflection_dir1 = at::zeros_like(w_h_3);
    Reflection<<<(width * height + thread - 1) / thread, thread>>>(intersection_pos.data<float>(),directions.data<float>(),intersect_nor.data<float>(),width, height,
               1, 1.4723,reflection_dir1.data<float>());

	at::Tensor inv_origins = at::zeros_like(w_h_3);
	GetInvOrigin << <(width * height + thread - 1) / thread, thread >> > (
		intersection_pos.data<float>(),
		refraction_dir1.data<float>(),
		width, height,
		bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, bounding_box_max_x, bounding_box_max_y, bounding_box_max_z,
		inv_origins.data<float>()
		);
//	cudaMemcpy(watch3, inv_origins.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);

	at::Tensor inv_voxel_position = at::zeros_like(w_h_3);
	at::Tensor inv_intersection_pos = at::zeros_like(w_h_3);
	Intersect << <(width * height + thread - 1) / thread, thread >> > (
		grid.data<float>(),
		inv_origins.data<float>(),
		refraction_dir1.data<float>(),
		origin_image_distances.data<float>(),
		pixel_distances.data<float>(),
		bounding_box_min_x,
		bounding_box_min_y,
		bounding_box_min_z,
		bounding_box_max_x,
		bounding_box_max_y,
		bounding_box_max_z,
		grid_res_x,
		grid_res_y,
		grid_res_z,
		inv_voxel_position.data<float>(),
		inv_intersection_pos.data<float>(),
		width,
		height,
		-1);
//	cudaMemcpy(watch3, inv_intersection_pos.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);


	at::Tensor inv_intersect_nor = at::zeros_like(w_h_3);
	IntersectNormal << <(width * height + thread - 1) / thread, thread >> > (
		grid.data<float>(),
		inv_intersection_pos.data<float>(),
		inv_voxel_position.data<float>(),
		width, height,
		(bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1),
		grid_res_x, grid_res_y, grid_res_y,
		bounding_box_min_x, bounding_box_min_y, bounding_box_min_z,
		inv_intersect_nor.data<float>(),
		-1
		);

	at::Tensor refraction_dir2 = at::zeros_like(w_h_3);
	at::Tensor attenuation2 = at::zeros_like(w_h);
	Refraction << <(width * height + thread - 1) / thread, thread >> > (
		inv_intersection_pos.data<float>(),
		refraction_dir1.data<float>(),
		inv_intersect_nor.data<float>(),
		width, height,
		1.4723, 1,
		refraction_dir2.data<float>(),
		attenuation2.data<float>(),
		total_reflect_mask.data<float>()
		);

//	cudaMemcpy(watch32, refraction_dir2.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);
//	for (int i = 0; i < 120000; ++i)
//	{
//		watch3[i] = watch32[i] - watch31[i];
//	}

	return { directions, intersection_pos, voxel_position, intersect_nor, reflection_dir1, refraction_dir1, inv_intersection_pos, inv_intersect_nor,refraction_dir2 };
}


std::vector<at::Tensor> ray_matching_cuda_cam(
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

//	float* watch = new float[200 * 200];
//	float* watch3 = new float[200 * 200 * 3];
//	float* watch31 = new float[120000];
//	float* watch32 = new float[120000];

    const int thread = 512;

    at::Tensor origins = at::zeros_like(w_h_3);
    at::Tensor directions = at::zeros_like(w_h_3);
    at::Tensor origin_image_distances = at::zeros_like(w_h);
    at::Tensor pixel_distances = at::zeros_like(w_h);


    GenerateRay2 << <(width * height + thread - 1) / thread, thread >> > (
            origins.data<float>(),
            directions.data<float>(),
            origin_image_distances.data<float>(),
            pixel_distances.data<float>(),
            width,
            height,
            fov,
            eye.data<float>(),
            lookat.data<float>(),
            lookup.data<float>());

//	cudaMemcpy(watch31, directions.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);

    at::Tensor voxel_position = at::zeros_like(w_h_3);
    at::Tensor intersection_pos = at::zeros_like(w_h_3);
    Intersect << <(width * height + thread - 1) / thread, thread >> > (
            grid.data<float>(),
                    origins.data<float>(),
                    directions.data<float>(),
                    origin_image_distances.data<float>(),
                    pixel_distances.data<float>(),
                    bounding_box_min_x,
                    bounding_box_min_y,
                    bounding_box_min_z,
                    bounding_box_max_x,
                    bounding_box_max_y,
                    bounding_box_max_z,
                    grid_res_x,
                    grid_res_y,
                    grid_res_z,
                    voxel_position.data<float>(),
                    intersection_pos.data<float>(),
                    width,
                    height,
                    1);

    at::Tensor intersect_nor = at::zeros_like(w_h_3);
    IntersectNormal << <(width * height + thread - 1) / thread, thread >> > (
            grid.data<float>(),
                    intersection_pos.data<float>(),
                    voxel_position.data<float>(),
                    width, height,
                    (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1),
                    grid_res_x, grid_res_y, grid_res_y,
                    bounding_box_min_x,bounding_box_min_y,bounding_box_min_z,
                    intersect_nor.data<float>(),
                    1
    );

    at::Tensor refraction_dir1 = at::zeros_like(w_h_3);
    at::Tensor attenuation1 = at::zeros_like(w_h);
    at::Tensor total_reflect_mask = at::zeros_like(w_h);
    Refraction << <(width * height + thread - 1) / thread, thread >> > (
            intersection_pos.data<float>(),
                    directions.data<float>(),
                    intersect_nor.data<float>(),
                    width, height,
                    1, 1.4723,
                    refraction_dir1.data<float>(),
                    attenuation1.data<float>(),
                    total_reflect_mask.data<float>()
    );

//	cudaMemcpy(watch3, refraction_dir1.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);
//	cudaMemcpy(watch, attenuation1.data_ptr, sizeof(float) * 200 * 200 * 1, cudaMemcpyDeviceToHost);
//	//cudaMemcpy(watch, total_reflect_mask.data_ptr, sizeof(float) * 200 * 200 * 1, cudaMemcpyDeviceToHost);

    at::Tensor reflection_dir1 = at::zeros_like(w_h_3);
    Reflection<<<(width * height + thread - 1) / thread, thread>>>(intersection_pos.data<float>(),directions.data<float>(),intersect_nor.data<float>(),width, height,
                                                                   1, 1.4723,reflection_dir1.data<float>());

    at::Tensor inv_origins = at::zeros_like(w_h_3);
    GetInvOrigin << <(width * height + thread - 1) / thread, thread >> > (
            intersection_pos.data<float>(),
                    refraction_dir1.data<float>(),
                    width, height,
                    bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, bounding_box_max_x, bounding_box_max_y, bounding_box_max_z,
                    inv_origins.data<float>()
    );
//	cudaMemcpy(watch3, inv_origins.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);

    at::Tensor inv_voxel_position = at::zeros_like(w_h_3);
    at::Tensor inv_intersection_pos = at::zeros_like(w_h_3);
    Intersect << <(width * height + thread - 1) / thread, thread >> > (
            grid.data<float>(),
                    inv_origins.data<float>(),
                    refraction_dir1.data<float>(),
                    origin_image_distances.data<float>(),
                    pixel_distances.data<float>(),
                    bounding_box_min_x,
                    bounding_box_min_y,
                    bounding_box_min_z,
                    bounding_box_max_x,
                    bounding_box_max_y,
                    bounding_box_max_z,
                    grid_res_x,
                    grid_res_y,
                    grid_res_z,
                    inv_voxel_position.data<float>(),
                    inv_intersection_pos.data<float>(),
                    width,
                    height,
                    -1);
//	cudaMemcpy(watch3, inv_intersection_pos.data_ptr, sizeof(float) * 200 * 200 * 3, cudaMemcpyDeviceToHost);


    at::Tensor inv_intersect_nor = at::zeros_like(w_h_3);
    IntersectNormal << <(width * height + thread - 1) / thread, thread >> > (
            grid.data<float>(),
                    inv_intersection_pos.data<float>(),
                    inv_voxel_position.data<float>(),
                    width, height,
                    (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1),
                    grid_res_x, grid_res_y, grid_res_y,
                    bounding_box_min_x, bounding_box_min_y, bounding_box_min_z,
                    inv_intersect_nor.data<float>(),
                    -1
    );

    at::Tensor refraction_dir2 = at::zeros_like(w_h_3);
    at::Tensor attenuation2 = at::zeros_like(w_h);
    Refraction << <(width * height + thread - 1) / thread, thread >> > (
            inv_intersection_pos.data<float>(),
                    refraction_dir1.data<float>(),
                    inv_intersect_nor.data<float>(),
                    width, height,
                    1.4723, 1,
                    refraction_dir2.data<float>(),
                    attenuation2.data<float>(),
                    total_reflect_mask.data<float>()
    );


    return {directions, intersection_pos, voxel_position,origin_image_distances,pixel_distances, refraction_dir2, reflection_dir1, attenuation1, attenuation2};
}

std::vector<at::Tensor> ray_matching_cuda_dir(
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


    const int thread = 512;
    at::Tensor voxel_position = at::zeros_like(w_h_3);
    at::Tensor intersection_pos = at::zeros_like(w_h_3);
    Intersect << <(width * height + thread - 1) / thread, thread >> > (
            grid.data<float>(),
                    origin.data<float>(),
                    direction.data<float>(),
                    origin_image_distances.data<float>(),
                    pixel_distances.data<float>(),
                    bounding_box_min_x,
                    bounding_box_min_y,
                    bounding_box_min_z,
                    bounding_box_max_x,
                    bounding_box_max_y,
                    bounding_box_max_z,
                    grid_res_x,
                    grid_res_y,
                    grid_res_z,
                    voxel_position.data<float>(),
                    intersection_pos.data<float>(),
                    width,
                    height,
                    1);



    return {intersection_pos, voxel_position};
}

//----------------------------------------------------------------------------------------------------------------------------------
__global__ void GenerateRayCamBS(
        float* origins,
        float* directions,
        float* origin_image_distances,
        float* pixel_distances,
        const int batchsize,
        const int width,
        const int height,
        const float fov,
        const float* eye, const float* lookat, const float* lookup)
{

    const int cam_index = blockIdx.z;
    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_index < width * height)
    {
        const float eye_x = eye[cam_index * 3 + 0];
        const float eye_y = eye[cam_index * 3 + 1];
        const float eye_z = eye[cam_index * 3 + 2];
        const float at_x = lookat[cam_index * 3 + 0];
        const float at_y = lookat[cam_index * 3 + 1];
        const float at_z = lookat[cam_index * 3 + 2];
        const float up_x = lookup[cam_index * 3 + 0];
        const float up_y = lookup[cam_index * 3 + 1];
        const float up_z = lookup[cam_index * 3 + 2];

        const float right = tan(DegToRad(fov));
        const float left = -right;
        const float top = (__int2float_rd(height) / __int2float_rd(width)) * right;
        const float bottom = -top;

        // Compute local base （eye为相机原点坐标，这一步计算相机坐标空间xyz轴的方向向量）
        const float w_x = -(eye_x - at_x) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
        const float w_y = -(eye_y - at_y) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
        const float w_z = -(eye_z - at_z) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
        const float cross_up_w_x = cross_x(up_x, up_y, up_z, w_x, w_y, w_z);
        const float cross_up_w_y = cross_y(up_x, up_y, up_z, w_x, w_y, w_z);
        const float cross_up_w_z = cross_z(up_x, up_y, up_z, w_x, w_y, w_z);
        const float u_x = (cross_up_w_x) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
        const float u_y = (cross_up_w_y) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
        const float u_z = (cross_up_w_z) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
        const float v_x = cross_x(w_x, w_y, w_z, u_x, u_y, u_z);
        const float v_y = cross_y(w_x, w_y, w_z, u_x, u_y, u_z);
        const float v_z = cross_z(w_x, w_y, w_z, u_x, u_y, u_z);



        const int x = pixel_index % width;
        const int y = pixel_index / width;
        const int i = 3 * pixel_index;

        // Compute point on view plane
        // Ray passes through the center of the pixel
        const float view_plane_x = left + (right - left) * (__int2float_rd(x) + 0.5) / __int2float_rd(width);
        const float view_plane_y = top - (top - bottom) * (__int2float_rd(y) + 0.5) / __int2float_rd(height);
        const float s_x = view_plane_x * u_x + view_plane_y * v_x + w_x;
        const float s_y = view_plane_x * u_y + view_plane_y * v_y + w_y;
        const float s_z = view_plane_x * u_z + view_plane_y * v_z + w_z;
        origins[cam_index * width * height * 3 + i] = eye_x;
        origins[cam_index * width * height * 3 + i + 1] = eye_y;
        origins[cam_index * width * height * 3 + i + 2] = eye_z;


        directions[cam_index * width * height * 3 + i] = s_x / length(s_x, s_y, s_z);
        directions[cam_index * width * height * 3 + i + 1] = s_y / length(s_x, s_y, s_z);
        directions[cam_index * width * height * 3 + i + 2] = s_z / length(s_x, s_y, s_z);

        origin_image_distances[cam_index * width * height + pixel_index] = length(s_x, s_y, s_z);
        pixel_distances[cam_index * width * height + pixel_index] = (right - left) / __int2float_rd(width);
    }




}

// Compute the intersection of the ray and the grid
// The intersection procedure uses ray marching to check if we have an interaction with the stored surface
// 添加了 让射线反向的inv, 正常inv为1，若反向则设置为-1
__global__ void IntersectBS(
        const float* grid,
        const float* origins,
        const float* directions,
        const float* origin_image_distances,
        const float* pixel_distances,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x,
        const int grid_res_y,
        const int grid_res_z,
        float* voxel_position,
        float* intersection_pos,
        const int batchsize,
        const int width,
        const int height,
        const int inv,
        const int error_code) {

    // Compute voxel size
    const float voxel_size = (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1);

    // Define constant values
    const int max_steps = 1000;
    bool first_time = true;
    float depth = 0;
    int gotten_result = 0;

    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int cam_index = blockIdx.z;
    const int cam_off = cam_index * width * height;

    if (pixel_index < width * height) {

        const int i = 3 * pixel_index;

        if (origins[cam_index * width * height * 3 + i] == error_code)
        {
            intersection_pos[cam_index * width * height * 3 + i] = error_code;
            intersection_pos[cam_index * width * height * 3 + i + 1] = error_code;
            intersection_pos[cam_index * width * height * 3 + i + 2] = error_code;
            voxel_position[cam_index * width * height * 3 + i] = error_code;
            voxel_position[cam_index * width * height * 3 + i + 1] = error_code;
            voxel_position[cam_index * width * height * 3 + i + 2] = error_code;
            return;
        }

        for (int steps = 0; steps < max_steps; steps++) {

            float reached_point_x = origins[cam_index * width * height * 3 + i] + inv * depth * directions[cam_index * width * height * 3 + i];
            float reached_point_y = origins[cam_index * width * height * 3 + i + 1] + inv * depth * directions[cam_off * 3 + i + 1];
            float reached_point_z = origins[cam_index * width * height * 3 + i + 2] + inv * depth * directions[cam_off * 3 + i + 2];

            // Get the signed distance value for the point the ray reaches
            const float distance = ValueAt(grid, reached_point_x, reached_point_y, reached_point_z,
                                           inv * directions[cam_off * 3 + i], inv * directions[cam_off * 3 + i + 1], inv * directions[cam_off * 3 + i + 2],
                                           bounding_box_min_x,
                                           bounding_box_min_y,
                                           bounding_box_min_z,
                                           bounding_box_max_x,
                                           bounding_box_max_y,
                                           bounding_box_max_z,
                                           grid_res_x,
                                           grid_res_y,
                                           grid_res_z, first_time);
            first_time = false;

            // Check if the ray is going ourside the bounding box
            if (distance == -1) {
                voxel_position[cam_off * 3 + i] = error_code;
                voxel_position[cam_off * 3 + i + 1] = error_code;
                voxel_position[cam_off * 3 + i + 2] = error_code;
                intersection_pos[cam_off * 3 + i] = error_code;
                intersection_pos[cam_off * 3 + i + 1] = error_code;
                intersection_pos[cam_off * 3 + i + 2] = error_code;
                gotten_result = 1;
                break;
            }

            // Check if we are close enough to the surface
            if (distance < pixel_distances[cam_off + pixel_index] / origin_image_distances[cam_off + pixel_index] * depth / 5) {
            //if (distance < 0) {

                // Compute the the minimum point of the intersecting voxel
                voxel_position[cam_off * 3 + i] = floorf((reached_point_x - bounding_box_min_x) / voxel_size);
                voxel_position[cam_off * 3 + i + 1] = floorf((reached_point_y - bounding_box_min_y) / voxel_size);
                voxel_position[cam_off * 3 + i + 2] = floorf((reached_point_z - bounding_box_min_z) / voxel_size);
                if (voxel_position[cam_off * 3 + i] == grid_res_x - 1) {
                    voxel_position[cam_off * 3 + i] = voxel_position[cam_off * 3 + i] - 1;
                }
                if (voxel_position[cam_off * 3 + i + 1] == grid_res_y - 1) {
                    voxel_position[cam_off * 3 + i + 1] = voxel_position[cam_off * 3 + i + 1] - 1;
                }
                if (voxel_position[cam_off * 3 + i + 2] == grid_res_z - 1) {
                    voxel_position[cam_off * 3 + i + 2] = voxel_position[cam_off * 3 + i + 2] - 1;
                }
                intersection_pos[cam_off * 3 + i] = reached_point_x;
                intersection_pos[cam_off * 3 + i + 1] = reached_point_y;
                intersection_pos[cam_off * 3 + i + 2] = reached_point_z;
                gotten_result = 1;
                break;
            }

            // Increase distance
            depth += distance;

        }

        if (gotten_result == 0) {

            // No intersections
            voxel_position[cam_off * 3 + i] = error_code;
            voxel_position[cam_off * 3 + i + 1] = error_code;
            voxel_position[cam_off * 3 + i + 2] = error_code;
            intersection_pos[cam_off * 3 + i] = error_code;
            intersection_pos[cam_off * 3 + i + 1] = error_code;
            intersection_pos[cam_off * 3 + i + 2] = error_code;
        }
    }
}

//batchsize个相机同时raymarching
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
        const int error_code)
{

    const int thread = 512/batchsize;

    at::Tensor origins = at::zeros_like(b_w_h_3);
    at::Tensor directions = at::zeros_like(b_w_h_3);
    at::Tensor origin_image_distances = at::zeros_like(b_w_h);
    at::Tensor pixel_distances = at::zeros_like(b_w_h);

    dim3 grid_dim((width * height + thread - 1)/thread, 1, batchsize);
    dim3 block_dim(thread, 1, 1);

    GenerateRayCamBS << < grid_dim, block_dim >> > (
            origins.data<float>(),
                    directions.data<float>(),
                    origin_image_distances.data<float>(),
                    pixel_distances.data<float>(),
                    batchsize,
                    width,
                    height,
                    fov,
                    eye.data<float>(),
                    lookat.data<float>(),
                    lookup.data<float>());

    at::Tensor voxel_position = at::zeros_like(b_w_h_3);
    at::Tensor intersection_pos = at::zeros_like(b_w_h_3);
    IntersectBS << <grid_dim, block_dim >> > (
                    grid.data<float>(),
                    origins.data<float>(),
                    directions.data<float>(),
                    origin_image_distances.data<float>(),
                    pixel_distances.data<float>(),
                    bounding_box_min_x,
                    bounding_box_min_y,
                    bounding_box_min_z,
                    bounding_box_max_x,
                    bounding_box_max_y,
                    bounding_box_max_z,
                    grid_res_x,
                    grid_res_y,
                    grid_res_z,
                    voxel_position.data<float>(),
                    intersection_pos.data<float>(),
                    batchsize,
                    width,
                    height,
                    1,
                    error_code);

    return { directions, intersection_pos, voxel_position,origin_image_distances,pixel_distances };
}


std::vector<at::Tensor> ray_matching_cuda_dir_bs(
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


    const int thread = 512/batchsize;
    dim3 grid_dim((width * height + thread - 1)/thread, 1, batchsize);
    dim3 block_dim(thread, 1, 1);

    at::Tensor voxel_position = at::zeros_like(b_w_h_3);
    at::Tensor intersection_pos = at::zeros_like(b_w_h_3);
    IntersectBS << <grid_dim, block_dim >> > (
                    grid.data<float>(),
                    origin.data<float>(),
                    direction.data<float>(),
                    origin_image_distances.data<float>(),
                    pixel_distances.data<float>(),
                    bounding_box_min_x,
                    bounding_box_min_y,
                    bounding_box_min_z,
                    bounding_box_max_x,
                    bounding_box_max_y,
                    bounding_box_max_z,
                    grid_res_x,
                    grid_res_y,
                    grid_res_z,
                    voxel_position.data<float>(),
                    intersection_pos.data<float>(),
                    batchsize,
                    width,
                    height,
                    1,
                            error_code);


    int a = 1;
    return { intersection_pos, voxel_position };
}



//---------------------------------------------------------------------------------------------------------------------



