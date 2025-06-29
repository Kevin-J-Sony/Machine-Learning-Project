#include "matrix_cuda.h"

#ifdef USE_CUDA

#ifdef ML_LIB_DEBUG_MODE
#define CHECK_CUDA_ERROR(cuda_op, msg) cudaError err = cuda_op; (err != cudaSuccess) { fprintf(stderr, "CUDA ERROR: %s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
#else
#define CHECK_CUDA_ERROR(cuda_op, msg) cudaError err = cuda_op;
#endif

__global__ void vec_add_kernel(number* out, const number* a, const number* b, size_t n) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		out[idx] = a[idx] + b[idx];
	}
}

__global__ void mat_add_kernel(number* out, const number* a, const number* b, size_t total_elems) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elems) {
		out[idx] = a[idx] + b[idx];
	}
}

__global__ void vec_sub_kernel(number* out, const number* a, const number* b, size_t n) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		out[idx] = a[idx] - b[idx];
	}
}

__global__ void mat_sub_kernel(number* out, const number* a, const number* b, size_t total_elems) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elems) {
		out[idx] = a[idx] - b[idx];
	}
}

__global__ void vec_scale_kernel(number* out, const number* in, number scale, size_t n) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		out[idx] = scale * in[idx];
	}
}

__global__ void mat_scale_kernel(number* out, const number* in, number scale, size_t total_elems) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elems) {
		out[idx] = scale * in[idx];
	}
}

__global__ void mat_mult_kernel(number* out, const number* a, const number* b,
								size_t a_rows, size_t a_cols, size_t b_cols) {
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < a_rows && col < b_cols) {
		number sum = 0.0f;
		for (size_t k = 0; k < a_cols; ++k) {
			sum += a[row * a_cols + k] * b[k * b_cols + col];
		}
		out[row * b_cols + col] = sum;
	}
}

__global__ void mat_entry_product_kernel(number* out, const number* a, const number* b, size_t total_elems) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elems) {
		out[idx] = a[idx] * b[idx];
	}
}



/* ********  WRAPPER FUNCTIONS ******** */



void vector_add_cuda(vector* out, const vector* a, const vector* b) {
	size_t n = a->size;
	number *d_out, *d_a, *d_b;

	CHECK_CUDA_ERROR(cudaMalloc(&d_out, n * sizeof(number)), "Failed to allocated (d_out) in vector_add_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_a, n * sizeof(number)), "Failed to allocated (d_a) in vector_add_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * sizeof(number)), "Failed to allocated (d_b) in vector_add_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_a, a->v, n * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (a) to (d_a) in vector_add_cuda");
	CHECK_CUDA_ERROR(cudaMemcpy(d_b, b->v, n * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (b) to (d_b) in vector_add_cuda");

	dim3 blockDim(256);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
	vec_add_kernel<<<gridDim, blockDim>>>(d_out, d_a, d_b, n);
	cudaDeviceSynchronize();

	CHECK_CUDA_ERROR(cudaMemcpy(out->v, d_out, n * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in vector_add_cuda");
	
	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in vector_add_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_a), "Failed to free (d_a) in vector_add_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_b), "Failed to free (d_b) in vector_add_cuda");
}

void matrix_add_cuda(matrix* out, const matrix* a, const matrix* b) {
	size_t total = a->number_of_rows * a->number_of_cols;
	number *d_out, *d_a, *d_b;

	CHECK_CUDA_ERROR(cudaMalloc(&d_out, total * sizeof(number)), "Failed to allocated (d_out) in matrix_add_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_a, total * sizeof(number)), "Failed to allocated (d_a) in matrix_add_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_b, total * sizeof(number)), "Failed to allocated (d_b) in matrix_add_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_a, a->m, total * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (a->m) to (d_a) in matrix_add_cuda");
	CHECK_CUDA_ERROR(cudaMemcpy(d_b, b->m, total * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (b->m) to (d_b) in matrix_add_cuda");
	
	dim3 blockDim(256);
	dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
	mat_add_kernel<<<gridDim, blockDim>>>(d_out, d_a, d_b, total);
	cudaDeviceSynchronize();

	CHECK_CUDA_ERROR(cudaMemcpy(out->m, d_out, total * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in matrix_add_cuda");

	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in matrix_add_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_a), "Failed to free (d_a) in matrix_add_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_b), "Failed to free (d_b) in matrix_add_cuda");
}

void vector_sub_cuda(vector* out, const vector* a, const vector* b) {
	size_t n = a->size;
	number *d_out, *d_a, *d_b;

	CHECK_CUDA_ERROR(cudaMalloc(&d_out, n * sizeof(number)), "Failed to allocated (d_out) in vector_sub_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_a, n * sizeof(number)), "Failed to allocated (d_a) in vector_sub_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * sizeof(number)), "Failed to allocated (d_b) in vector_sub_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_a, a->v, n * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (a) to (d_a) in vector_sub_cuda");
	CHECK_CUDA_ERROR(cudaMemcpy(d_b, b->v, n * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (b) to (d_b) in vector_sub_cuda");

	dim3 blockDim(256);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
	vec_sub_kernel<<<gridDim, blockDim>>>(d_out, d_a, d_b, n);
	cudaDeviceSynchronize();

	CHECK_CUDA_ERROR(cudaMemcpy(out->v, d_out, n * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in vector_sub_cuda");
	
	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in vector_sub_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_a), "Failed to free (d_a) in vector_sub_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_b), "Failed to free (d_b) in vector_sub_cuda");
}

void matrix_sub_cuda(matrix* out, const matrix* a, const matrix* b) {
	size_t total = a->number_of_rows * a->number_of_cols;
	number *d_out, *d_a, *d_b;

	CHECK_CUDA_ERROR(cudaMalloc(&d_out, total * sizeof(number)), "Failed to allocated (d_out) in matrix_sub_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_a, total * sizeof(number)), "Failed to allocated (d_a) in matrix_sub_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_b, total * sizeof(number)), "Failed to allocated (d_b) in matrix_sub_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_a, a->m, total * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (a->m) to (d_a) in matrix_sub_cuda");
	CHECK_CUDA_ERROR(cudaMemcpy(d_b, b->m, total * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (b->m) to (d_b) in matrix_sub_cuda");
	
	dim3 blockDim(256);
	dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
	mat_sub_kernel<<<gridDim, blockDim>>>(d_out, d_a, d_b, total);
	cudaDeviceSynchronize();

	CHECK_CUDA_ERROR(cudaMemcpy(out->m, d_out, total * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in matrix_sub_cuda");

	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in matrix_sub_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_a), "Failed to free (d_a) in matrix_sub_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_b), "Failed to free (d_b) in matrix_sub_cuda");
}

void vector_scale_cuda(vector* out, const vector* in, const number scale) {
	size_t n = a->size;
	number *d_out, *d_in;

	CHECK_CUDA_ERROR(cudaMalloc(&d_out, n * sizeof(number)), "Failed to allocated (d_out) in vector_scale_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_in, n * sizeof(number)), "Failed to allocated (d_in) in vector_scale_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_in, a->v, n * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (in) to (d_in) in vector_scale_cuda");

	dim3 blockDim(256);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
	vec_scale_kernel<<<gridDim, blockDim>>>(d_out, d_in, scale, n);
	cudaDeviceSynchronize();

	CHECK_CUDA_ERROR(cudaMemcpy(out->v, d_out, n * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in vector_scale_cuda");
	
	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in vector_scale_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_in), "Failed to free (d_in) in vector_scale_cuda");
}

void matrix_scale_cuda(matrix* out, const matrix* in, const number scale) {
	size_t total = a->number_of_rows * a->number_of_cols;
	number *d_out, *d_in;

	CHECK_CUDA_ERROR(cudaMalloc(&d_out, total * sizeof(number)), "Failed to allocated (d_out) in matrix_scale_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_in, total * sizeof(number)), "Failed to allocated (d_in) in matrix_scale_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_in, in->m, total * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (in) to (d_in) in matrix_scale_cuda");
	
	dim3 blockDim(256);
	dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
	mat_scale_kernel<<<gridDim, blockDim>>>(d_out, d_in, scale, total);
	cudaDeviceSynchronize();

	CHECK_CUDA_ERROR(cudaMemcpy(out->m, d_out, total * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in matrix_sub_cuda");

	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in matrix_sub_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_in), "Failed to free (d_a) in matrix_sub_cuda");
}


void matrix_mult_cuda(matrix* out, const matrix* a, const matrix* b) {
	size_t m = a->number_of_rows;
	size_t p = a->number_of_cols;
	size_t n = b->number_of_cols;

	number *d_out, *d_a, *d_b;
	CHECK_CUDA_ERROR(cudaMalloc(&d_out, m * n * sizeof(number)), "Failed to allocated (d_out) in matrix_mult_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_a, m * p * sizeof(number)), "Failed to allocated (d_a) in matrix_mult_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_b, p * n * sizeof(number)), "Failed to allocated (d_b) in matrix_mult_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_a, a->m, m * p * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (a) to (d_a) in matrix_mult_cuda");
	CHECK_CUDA_ERROR(cudaMemcpy(d_b, b->m, p * n * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (b) to (d_b) in matrix_mult_cuda");
	
	dim3 blockDim(16,16);
	dim3 gridDim((n + threads.x - 1)/threads.x,
				(m + threads.y - 1)/threads.y);
	mat_mult_kernel<<<gridDim, blockDim>>>(d_out, d_a, d_b, m, p, n);
	
	CHECK_CUDA_ERROR(cudaMemcpy(out->m, d_out, m * n * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in matrix_mult_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in matrix_mult_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_a), "Failed to free (d_a) in matrix_mult_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_b), "Failed to free (d_b) in matrix_mult_cuda");
}

void matrix_entrywise_product_cuda(matrix* out, const matrix* a, const matrix* b) {
	size_t total = a->number_of_rows * a->number_of_cols;
	number *d_out, *d_a, *d_b;

	CHECK_CUDA_ERROR(cudaMalloc(&d_out, total * sizeof(number)), "Failed to allocated (d_out) in matrix_entrywise_product_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_a, total * sizeof(number)), "Failed to allocated (d_a) in matrix_entrywise_product_cuda");
	CHECK_CUDA_ERROR(cudaMalloc(&d_b, total * sizeof(number)), "Failed to allocated (d_b) in matrix_entrywise_product_cuda");
	
	CHECK_CUDA_ERROR(cudaMemcpy(d_a, a->m, total * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (a->m) to (d_a) in matrix_entrywise_product_cuda");
	CHECK_CUDA_ERROR(cudaMemcpy(d_b, b->m, total * sizeof(number), cudaMemcpyHostToDevice), "Failed to copy contents of (b->m) to (d_b) in matrix_entrywise_product_cuda");
	
	dim3 blockDim(256);
	dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
	mat_entry_product_kernel<<<gridDim, blockDim>>>(d_out, d_a, d_b, total);
	cudaDeviceSynchronize();

	CHECK_CUDA_ERROR(cudaMemcpy(out->m, d_out, total * sizeof(number), cudaMemcpyDeviceToHost), "Failed to copy contents of (d_out) to (out) in matrix_entrywise_product_cuda");

	CHECK_CUDA_ERROR(cudaFree(d_out), "Failed to free (d_out) in matrix_entrywise_product_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_a), "Failed to free (d_a) in matrix_entrywise_product_cuda");
	CHECK_CUDA_ERROR(cudaFree(d_b), "Failed to free (d_b) in matrix_entrywise_product_cuda");	
}

#endif
