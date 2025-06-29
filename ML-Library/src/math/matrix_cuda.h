#include "../mllib.h"

#ifndef MLLIB_MATRIX_CUDA_H
#define MLLIB_MATRIX_CUDA_H

#ifdef USE_CUDA
#include <cuda_runtime.h>

__global__ void vec_add_kernel(number* out, const number* a, const number* b, size_t n);
__global__ void mat_add_kernel(number* out, const number* a, const number* b, size_t total_elems);
__global__ void mat_mult_kernel(number* out, const number* a, const number* b, size_t a_rows, size_t a_cols, size_t b_cols);

__global__ void vec_add_kernel(number* out, const number* a, const number* b, size_t n);
__global__ void mat_add_kernel(number* out, const number* a, const number* b, size_t total_elems);
__global__ void vec_sub_kernel(number* out, const number* a, const number* b, size_t n);
__global__ void mat_sub_kernel(number* out, const number* a, const number* b, size_t total_elems);
__global__ void vec_scale_kernel(number* out, const number* in, number scale, size_t n);
__global__ void mat_scale_kernel(number* out, const number* in, number scale, size_t total_elems);

__global__ void mat_mult_kernel(number* out, const number* a, const number* b, size_t a_rows, size_t a_cols, size_t b_cols);
__global__ void mat_entry_product_kernel(number* out, const number* a, const number* b, size_t total_elems);



void vector_add_cuda(vector* out, const vector* a, const vector* b);
void matrix_add_cuda(matrix* out, const matrix* a, const matrix* b);

void vector_sub_cuda(vector* out, const vector* a, const vector* b);
void matrix_sub_cuda(matrix* out, const matrix* a, const matrix* b);

void vector_scale_cuda(vector* out, const vector* in, const number scale);
void matrix_scale_cuda(matrix* out, const matrix* in, const number scale);

void matrix_mult_cuda(matrix* out, const matrix* a, const matrix* b);
void matrix_entrywise_product_cuda(matrix* out, const matrix* a, const matrix* b);


#endif
#endif
