// Simple functions to enhance functionality
// When finished, use malloc rather than calloc
#include "matrix.h"

vector* init_vec(size_t s) {
	vector* vec;
	vec = (vector *)malloc(sizeof(vector));
	vec->size = s;	
	vec->v = (number *)malloc(s * sizeof(number));	
	return vec;
}


void del_vec(vector* vec) {
	free(vec->v);
	free(vec);
}

matrix* init_mat(size_t nrows, size_t ncols) {
	matrix* mat;
	mat = (matrix *)malloc(sizeof(matrix));
	mat->number_of_rows = nrows;
	mat->number_of_cols = ncols;
	mat->m = (number *)malloc(nrows * ncols * sizeof(number));	
	return mat;
}

void del_mat(matrix* mat) {
	free(mat->m);
	free(mat);
}


// Add two vectors of the same size together and store it in output.
void vector_add(vector* out, const vector* a, const vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->size == b->size && a->size == out->size) ) {
		fprintf(stderr, "ERROR IN VECTOR ADDITION: Size mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifndef USE_CUDA
	for (int i = 0; i < a->size; i++) {
		out->v[i] = a->v[i] + b->v[i];
	}
	#else
	vector_add_cuda(out, a, b);
	#endif
}

// Add two matrices of the same dimensions together and store it in output
void matrix_add(matrix* out, const matrix* a, const matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->number_of_rows == b->number_of_rows && a->number_of_rows == out->number_of_rows) ||
		! (a->number_of_cols == b->number_of_cols && a->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX ADDITION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifndef USE_CUDA
	for (int i = 0; i < a->number_of_rows; i++) {
		for (int j = 0; j < a->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(a, i, j) + VALUE_AT(b, i, j);
		}
	}
	#else
	matrix_add_cuda(out, a, b);
	#endif
}

// Subtract vector b from vector a and store it in vector out
void vector_sub(vector* out, const vector* a, const vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->size == b->size && a->size == out->size) ) {
		fprintf(stderr, "ERROR IN VECTOR SUBTRACTION: Size mismatch\n");
		exit(EXIT_FAILURE);
		// free to exit since when process ends, the virtual address space also is also terminated
		// however, unsure of the situation when dealing with cuda
	}
	#endif

	#ifndef USE_CUDA
	for (int i = 0; i < a->size; i++) {
		out->v[i] = a->v[i] - b->v[i];
	}
	#else
	vector_sub_cuda(out, a, b);
	#endif
}

// Subtract matrix b from matrix a and store it in out
void matrix_sub(matrix* out, const matrix* a, const matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (! (a->number_of_rows == b->number_of_rows && a->number_of_rows == out->number_of_rows) ||
		! (a->number_of_cols == b->number_of_cols && a->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX SUBTRACTION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifndef USE_CUDA
	for (int i = 0; i < a->number_of_rows; i++) {
		for (int j = 0; j < a->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(a, i, j) - VALUE_AT(b, i, j);
		}
	}
	#else
	matrix_sub_cuda(out, a, b);
	#endif
}

// Scale every entry in the input vector and store it in the output
void vector_scale(vector* out, const vector* in, const number scale) {
	#ifdef ML_LIB_DEBUG_MODE
	if (out->size != in->size) {
		fprintf(stderr, "ERROR IN VECTOR SCALE: Input/Output size mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifndef USE_CUDA
	for (int i = 0; i < in->size; i++) {
		out->v[i] = scale * in->v[i];
	}
	#else
	vector_scale_cuda(out, in, scale);
	#endif
}

// Scale every entry in the input matrix and store it in the output
void matrix_scale(matrix* out, const matrix* in, const number scale) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((out->number_of_rows != in->number_of_rows) || (out->number_of_cols != in->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX SCALE: Input/Output dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifndef USE_CUDA	
	for (int i = 0; i < out->number_of_rows; i++) {
		for (int j = 0; j < out->number_of_cols; j++) {
			VALUE_AT(out, i, j) = scale * VALUE_AT(in, i, j);
		}
	}
	#else
	matrix_scale_cuda(out, in, scale);
	#endif
}

// Your standard matrix multiplication. Multiply a and b and store it in out
void matrix_mult(matrix* out, const matrix* a, const matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	// Recall that matrix multiplication is valid only when a is (m, p) and b is (p, n)
	// The resulting output is (m, n)
	if (! (a->number_of_cols == b->number_of_rows && a->number_of_rows == out->number_of_rows
			&& b->number_of_cols == out->number_of_cols) ) {
		fprintf(stderr, "ERROR IN MATRIX MULTIPLICATION: Dimension mismatch\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifndef USE_CUDA
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
	#else
	matrix_mult_cuda(out, a, b);
	#endif
}


// Multiply every entry in the first input with the corresponding entry in the second input and store it in the corresponding entry in the output matrix
void matrix_entrywise_product(matrix* out, const matrix* product_one, const matrix* product_two) {
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

	#ifndef USE_CUDA
	for (int i = 0; i < out->number_of_rows; i++) {
		for (int j = 0; j < out->number_of_cols; j++) {
			VALUE_AT(out, i, j) = VALUE_AT(product_one, i, j) * VALUE_AT(product_two, i, j);
		}
	}
	#else
	matrix_entrywise_product_cuda(out, product_one, product_two);
	#endif
}


// For each column of the input matrix, add the vector to it and store the corresponding output in another matrix
void add_vector_to_matrix(matrix* out, const matrix* mat, const vector* vec) {
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


// Sum up all the columns in the input matrix and store it in the output vector
void matrix_col_sum(vector* out, const matrix* in) {
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


// Store transpose of input into output
void matrix_transpose(matrix* out, const matrix* in) {
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


// Copy matrix from input to output
void copy_matrix(matrix* out, const matrix* in) {
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



// Basic debug functions
#ifdef ML_LIB_DEBUG_MODE
void print_mat(matrix* mat) {
	size_t nrows = mat->number_of_rows;
	size_t ncols = mat->number_of_cols;
	number* m = mat->m;

	fprintf(stdout, "----------\nMatrix info\n");
	fprintf(stdout, "Number of rows: %lu \t Number of columns: %lu\n", nrows, ncols);

	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			fprintf(stdout, "%f ", mat->m[i * ncols + j]);
		}
		fprintf(stdout, "\n");
	}
}

void print_vec(vector* vec) {
	size_t size = vec->size;
	number* v = vec->v;

	fprintf(stdout, "----------\nVector info\n");
	fprintf(stdout, "Size of vector: %lu \n", size);

	for (int i = 0; i < size; i++) {
		fprintf(stdout, "%f ", v[i]);
	}
	fprintf(stdout, "\n");
}

#endif