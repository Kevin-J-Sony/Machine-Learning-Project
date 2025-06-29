#include "../mllib.h"


#ifndef MLLIB_MATRIX_H
#define MLLIB_MATRIX_H

#ifdef USE_CUDA
#include "matrix_cuda.h"
#endif


// define the basic vector data structures used for this project
// since we are not using c++, we need to define vectors and matrices
struct vector_ {
	number* v;
	size_t size;
};
typedef struct vector_ vector;

struct matrix_ {
	number* m;
	size_t number_of_rows;
	size_t number_of_cols;
};
typedef struct matrix_ matrix;

#define VALUE_AT(mat, i, j) mat->m[i * mat->number_of_cols + j]

// basic data structure functions
vector* init_vec(size_t size);
void del_vec(vector* mat);

matrix* init_mat(size_t nrows, size_t ncols);
void del_mat(matrix* mat);

// basic math functions required
void vector_add(vector* out, const vector* a, const vector* b);
void matrix_add(matrix* out, const matrix* a, const matrix* b);
void vector_sub(vector* out, const vector* a, const vector* b);
void matrix_sub(matrix* out, const matrix* a, const matrix* b);

void vector_scale(vector* out, const vector* in, const number scale);
void matrix_scale(matrix* out, const matrix* in, const number scale);

void matrix_mult(matrix* out, const matrix* a, const matrix* b);
void matrix_entrywise_product(matrix* out, const matrix* product_one, const matrix* product_two);


// Functions which should not be implemented in CUDA

// matrix-vector operations
void add_vector_to_matrix(matrix* out, const matrix* mat, const vector* vec);
void matrix_col_sum(vector* out, const matrix* in);


// basic matrix operations
void matrix_transpose(matrix* out, const matrix* in);
void copy_matrix(matrix* out, const matrix* in);



#ifdef ML_LIB_DEBUG_MODE
void print_mat(matrix* mat);
void print_vec(vector* vec);
#endif

#endif
