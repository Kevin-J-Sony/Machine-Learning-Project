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

matrix* init_mat(size_t nrows, size_t ncols) {
	matrix* mat;
	#ifdef ML_LIB_DEBUG_MODE
	mat = (matrix *)calloc(1, sizeof(matrix));
	#else
	mat = (matrix *)malloc(sizeof(matrix));
	#endif

	mat->number_of_rows = nrows;
	mat->number_of_cols = ncols;
	
	#ifdef ML_LIB_DEBUG_MODE
	mat->m = (number *)calloc(nrows * ncols, sizeof(number));
	#else
	mat->m = (number *)malloc(nrows * ncols * sizeof(number));
	#endif
	
	return mat;
}

void del_mat(matrix* mat) {
	free(mat->m);
	free(mat);
}

/* *** General vector, matrix operations *** */


/**
 * Add two vectors of the same size together and store it in output.
 */
void vector_add(vector* out, vector* a, vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->size == b->size && a->size == out->size) ) {
		fprintf(stderr, "ERROR IN VECTOR ADDITION: Size mismatch\n");
		exit(EXIT_FAILURE);
		// free to exit since when process ends, the virtual address space also is also terminated
		// however, unsure of the situation when dealing with cuda
	}
	#endif
	for (int i = 0; i < a->size; i++) {
		out->v[i] = a->v[i] + b->v[i];
	}
}

/**
 * Add two matrices of the same dimensions together and store it in output
 */
void matrix_add(matrix* out, matrix* a, matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->number_of_rows == b->number_of_rows && a->number_of_rows == out->number_of_rows) ||
		! (a->number_of_cols == b->number_of_cols && a->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX ADDITION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < a->number_of_rows; i++) {
		for (int j = 0; j < a->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(a, i, j) + VALUE_AT(b, i, j);
			// out->m[i * ncols + j] = a->m[i * ncols + j] + b->m[i * ncols + j];
		}
	}
}

void vector_sub(vector* out, vector* a, vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->size == b->size && a->size == out->size) ) {
		fprintf(stderr, "ERROR IN VECTOR SUBTRACTION: Size mismatch\n");
		exit(EXIT_FAILURE);
		// free to exit since when process ends, the virtual address space also is also terminated
		// however, unsure of the situation when dealing with cuda
	}
	#endif
	for (int i = 0; i < a->size; i++) {
		out->v[i] = a->v[i] - b->v[i];
	}
}

void matrix_sub(matrix* out, matrix* a, matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->number_of_rows == b->number_of_rows && a->number_of_rows == out->number_of_rows) ||
		! (a->number_of_cols == b->number_of_cols && a->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX SUBTRACTION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < a->number_of_rows; i++) {
		for (int j = 0; j < a->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(a, i, j) - VALUE_AT(b, i, j);
			// out->m[i * ncols + j] = a->m[i * ncols + j] - b->m[i * ncols + j];
		}
	}
}


void vector_scale(vector* out, vector* in, number scale) {
	#ifdef ML_LIB_DEBUG_MODE
	if (out->size != in->size) {
		fprintf(stderr, "ERROR IN VECTOR SCALE: Input/Output size mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif
	for (int i = 0; i < in->size; i++) {
		out->v[i] = scale * in->v[i];
	}
}

void matrix_scale(matrix* out, matrix* in, number scale) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((out->number_of_rows != in->number_of_rows) || (out->number_of_cols != in->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX SCALE: Input/Output dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < out->number_of_rows; i++) {
		for (int j = 0; j < out->number_of_cols; j++) {
			VALUE_AT(out, i, j) = scale * VALUE_AT(in, i, j);
		}
	}
}

/**
 * Basic matrix multiplication.
 */
void matrix_mult(matrix* out, matrix* a, matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	// Recall that matrix multiplication is valid only when a is (m, p) and b is (p, n)
	// The resulting output is (m, n)
	if (! (a->number_of_cols == b->number_of_rows && a->number_of_rows == out->number_of_rows
			&& b->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX MULTIPLICATION: Dimension mismatch\n");
		fprintf(stderr, "out: (%lu x %lu)\n", out->number_of_rows, out->number_of_cols);
		fprintf(stderr, "a: (%lu x %lu)\n", a->number_of_rows, a->number_of_cols);
		fprintf(stderr, "b: (%lu x %lu)\n", b->number_of_rows, b->number_of_cols);
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < out->number_of_rows; i++) {
		for (int j = 0; j < out->number_of_cols; j++) {
			VALUE_AT(out, i, j) = 0;
			// out->m[i * n + j] = 0;
			for (int k = 0; k < a->number_of_cols; k++) {
				VALUE_AT(out, i, j) += VALUE_AT(a, i, k) * VALUE_AT(b, k, j);
				// out->m[i * n + j] += a->m[i * p + k] * b->m[k * n + j];
			}
		}
	}

}

/**
 * Apply a matrix transformation to a vector. A matrix is simply a linear transformation \matbb{R}^n -> \matbb{R}^m
 */
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
			out->v[i] += VALUE_AT(a, i, j) * b->v[j];
		}
	}
}

/**
 * For each column of the input matrix, add the vector to it and store the corresponding output in another matrix
 */
void add_vector_to_matrix(matrix* out, matrix* mat, vector* vec) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((mat->number_of_cols != out->number_of_cols) || (mat->number_of_rows != out->number_of_rows)) {
		fprintf(stderr, "ERROR IN ADDITION OF VECTORS TO COLUMNS OF MATRIX MATRIX: Dimensions of input and output matrices do not match.\n");
		exit(EXIT_FAILURE);
	}
	
	if (mat->number_of_rows != vec->size) {
		fprintf(stderr, "ERROR IN ADDITION OF VECTORS TO COLUMNS OF MATRIX MATRIX: The number of rows doesn't equal the number of entries in the vector.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < vec->size; i++) {
		for (int col = 0; col < mat->number_of_cols; col++) {
			VALUE_AT(out, i, col) = VALUE_AT(mat, i, col) + vec->v[i];
		}
	}
}

void matrix_entrywise_product(matrix* out, matrix* product_one, matrix* product_two) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (product_one->number_of_rows != product_two->number_of_rows) || 
		 (product_one->number_of_cols != product_two->number_of_cols)) {
		fprintf(stderr, "ERROR IN MATRIX ENTRYWISE PRODUCT: Dimensions of inputs do not match.\n");
		exit(EXIT_FAILURE);
	}

	if ( (product_one->number_of_rows != out->number_of_rows) || 
		 (product_one->number_of_cols != out->number_of_cols)) {
		fprintf(stderr, "ERROR IN MATRIX ENTRYWISE PRODUCT: Dimensions of output does not match dimensions of inputs.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < out->number_of_rows; i++) {
		for (int j = 0; j < out->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(product_one, i, j) * VALUE_AT(product_two, i, j);
			// out->m[i * out->number_of_cols + j] = product_one->m[i * out->number_of_cols + j] + product_two->m[i * out->number_of_cols + j];
		}
	}
}


void matrix_transpose(matrix* out, matrix* in) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((out->number_of_cols != in->number_of_rows) || (out->number_of_rows != in->number_of_cols)) {
		fprintf(stderr, "ERROR IN MATRIX TRANSPOSE: Dimensions of output and input matrices do not correlate.\n");
		exit(EXIT_FAILURE);
	}
	#endif
	for (int i = 0; i < out->number_of_rows; i++) {
		for (int j = 0; j < out->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(in, j, i);
		}
	}
}

void matrix_col_sum(vector* out, matrix* in) {
	#ifdef ML_LIB_DEBUG_MODE
	if (out->size != in->number_of_rows) {
		fprintf(stderr, "ERROR IN COLUMN SUM OF MATRIX: Size of vector does not match column length of matrix.\n");
		exit(EXIT_FAILURE);
	}
	#endif
	for (int i = 0; i < out->size; i++) {
		out->v[i] = 0;
		for (int j = 0; j < in->number_of_cols; j++) {
			out->v[i] += VALUE_AT(in, i, j);
		}
	}
}


/**
 * Copy matrix from input to output
 */
void copy_matrix(matrix* out, matrix* in) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((out->number_of_cols != in->number_of_cols) || (out->number_of_rows != in->number_of_rows)) {
		fprintf(stderr, "ERROR IN COPYING MATRIX: Dimensions of output and input matrices do not match.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < out->number_of_rows; i++) {
		for (int j = 0; j < out->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(in, i, j);
		}
	}
}
