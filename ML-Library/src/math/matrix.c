// Simple functions to enhance functionality
// When finished, use malloc rather than calloc
#include "matrix.h"

vector* init_vec(size_t s) {
	vector* vec;
	#ifdef ML_LIB_DEBUG_MODE
	vec = (vector *)calloc(1, sizeof(vector));
	#else
	vec = (vector *)malloc(sizeof(vector));
	#endif

	vec->size = s;
	
	#ifdef ML_LIB_DEBUG_MODE
	vec->v = (number *)calloc(s, sizeof(number));
	#else
	vec->v = (number *)malloc(s * sizeof(number));
	#endif
	
	return vec;
}


void del_vec(vector* vec) {
	free(vec->v);
	free(vec);
}

matrix* init_mat(size_t row, size_t col) {
	matrix* mat;
	#ifdef ML_LIB_DEBUG_MODE
	mat = (matrix *)calloc(1, sizeof(matrix));
	#else
	mat = (matrix *)malloc(sizeof(matrix));
	#endif

	mat->number_of_rows = row;
	mat->number_of_cols = col;
		
	#ifdef ML_LIB_DEBUG_MODE
	mat->m = (number *)calloc(row * col, sizeof(number));
	#else
	mat->m = (number *)malloc(row * col * sizeof(number));
	#endif
	
	return mat;
}

void del_mat(matrix* mat) {
	free(mat->m);
	free(mat);
}

/* *** General vector, matrix operations *** */


void vector_add(vector* out, vector* a, vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->size == b->size && a->size == out->size) ) {
		fprintf(stderr, "ERROR IN VECTOR ADDITION: Size mismatch\n");
		exit(EXIT_FAILURE);
		// free to exit since when process ends, the virtual address space also is also terminated
		// however, unsure of the situation when dealing with cuda
	}
	#endif
	int n = a->size;
	for (int i = 0; i < n; i++) {
		out->v[i] = a->v[i] + b->v[i];
	}
}

void matrix_add(matrix* out, matrix* a, matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->number_of_rows == b->number_of_rows && a->number_of_rows == out->number_of_rows) ||
		! (a->number_of_cols == b->number_of_cols && a->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX ADDITION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif
	size_t nrows = a->number_of_rows;
	size_t ncols = a->number_of_cols;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			out->m[i * ncols + j] = a->m[i * ncols + j] + b->m[i * ncols + j];
		}
	}
}

void matrix_mult(matrix* out, matrix* a, matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	// Recall that matrix multiplication is valid only when a is (m, p) and b is (p, n)
	// The resulting output is (m, n)
	if (! (a->number_of_cols == b->number_of_rows && a->number_of_rows == out->number_of_rows
			&& b->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX MULTIPLICATION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	size_t m = a->number_of_rows; // = out->number_of_rows
	size_t n = b->number_of_cols; // = out->number_of_cols
	size_t p = a->number_of_cols; // = b->number_of_rows

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			out->m[i * n + j] = 0;
			for (int k = 0; k < p; k++) {
				out->m[i * n + j] += a->m[i * p + k] * b->m[k * n + j];
			}
		}
	}

}

void matrix_vector_mult(vector* out, matrix* a, vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->number_of_cols == b->size && a->number_of_rows == out->size) ) {
		fprintf(stderr, "ERROR IN MATRIX VECTOR MULTIPLICATION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < out->size; i++) {
		out->v[i] = 0;
		for (int j = 0; j < b->size; j++) {
			out->v[i] += a->m[i * a->number_of_cols + j] * b->v[j];
		}
	}
}
