#include "../mllib.h"


#ifndef MLLIB_MATRIX_H
#define MLLIB_MATRIX_H

// define a data type 'number' to take on float, double, or long double
typedef float number;

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

// basic data structure functions
vector* init_vec(size_t size);
void del_vec(vector* mat);

matrix* init_mat(size_t row, size_t col);
void del_mat(matrix* mat);

// basic math functions required
void vector_add(vector* out, vector* a, vector* b);
void matrix_add(matrix* out, matrix* a, matrix* b);
void vector_scale(vector* out, vector* in, number scale);
void matrix_scale(matrix* out, matrix* in, number scale);

void matrix_mult(matrix* out, matrix* a, matrix* b);
void matrix_vector_mult(vector* out, matrix* a, vector* b);
void add_vector_to_matrix(matrix* out, matrix* mat, vector* vec);
void matrix_entrywise_product(matrix* out, matrix* product_one, matrix* product_two);

#endif
