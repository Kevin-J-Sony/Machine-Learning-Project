// Simple functions to enhance functionality
// When finished, use malloc rather than calloc
#include "matrix.h"

/**
 * @brief Return a vector with size 'length'
 * 
 * @param length size of te vector
 * @return pointer to vector of size 'length'
 */
vector* init_vec(int length) {
	vector* vec = (vector *)calloc(1, sizeof(struct vector_));

	#ifdef ML_LIB_DEBUG_MODE
	if ((void *)vec == NULL) {
		fprintf(stderr, "Failed vector initialization\n");
		exit(1);
	}
	#endif

	vec->size = length;
	vec->vector = (number *)calloc(length, sizeof(number));

	#ifdef ML_LIB_DEBUG_MODE
	if ((void *)vec->vector == NULL) {
		fprintf(stderr, "Failed vector allocation\n");
		exit(1);
	}
	#endif

	return vec;
}

/**
 * @brief Return a matrix of dimensions 'row' by 'col'
 * 
 * @param row 
 * @param col 
 * @return pointer to matrix of size 'row' by 'col'
 */
matrix* init_mat(int row, int col) {
	matrix* mat = (matrix *)calloc(1, sizeof(struct matrix_));

	#ifdef ML_LIB_DEBUG_MODE
	if ((void *)mat == NULL) {
		fprintf(stderr, "Failed matrix initialization\n");
		exit(1);
	}
	#endif

	mat->row = row;
	mat->col = col;
	mat->matrix = (number **)calloc(row, sizeof(number *));

	#ifdef ML_LIB_DEBUG_MODE
	if ((void *)mat->matrix == NULL) {
		fprintf(stderr, "Failed matrix column allocation\n");
		exit(1);
	}
	#endif

	for (int i = 0; i < row; i++) {
		mat->matrix[i] = (number *)calloc(col, sizeof(number));

		#ifdef ML_LIB_DEBUG_MODE
		if ((void *)mat->matrix[i] == NULL) {
			fprintf(stderr, "Failed matrix row allocation\n");
			exit(1);
		}
		#endif

}
	return mat;
}

// Functions to deallocate vectors and matrices
/**
 * @brief Deallocates vector
 * 
 * @param v vector to be deallocated
 */
void free_vec(vector* v) {
	free(v->vector);
	free(v);
}

/**
 * @brief Deallocates matrix
 * 
 * @param m matrix to be deallocated
 */
void free_mat(matrix* m) {
	for (int i = 0; i < m->row; i++) {
		free(m->matrix[i]);
	}
	free(m->matrix);
	free(m);
}

// Get functions
/**
 * @brief Return value of vector at index
 * 
 * @param v 
 * @param index 
 * @return number 
 */
number get_vec(vector* v, int index) {
	return v->vector[index];
}

/**
 * @brief Return value of matrix at given row and column
 * 
 * @param m 
 * @param row_index 
 * @param col_index 
 * @return number 
 */
number get_mat(matrix* m, int row_index, int col_index) {
	return m->matrix[row_index][col_index];
}

// Set functions
/**
 * @brief Set value of vector at index
 * 
 * @param v 
 * @param index 
 * @param value 
 */
void set_vec(vector* v, int index, int value) {
	v->vector[index] = value;
}

/**
 * @brief Set value of matrix at given row and column
 * 
 * @param m 
 * @param row_index 
 * @param col_index 
 * @param value 
 */
void set_mat(matrix* m, int row_index, int col_index, int value) {
	(m->matrix[row_index])[col_index] = value;
}

// Functions for basic operation
vector* add(vector* a, vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (a->size != b->size) {
		fprintf(stderr, "Vector addition with different vector lengths not allowed\n");
		exit(1);
	}
	#endif
	int n = a->size;
	vector* c = init_vec(n);
	for (int idx = 0; idx < n; idx++) {
		set(c, idx, get(a, idx) + get(b, idx))
	}
	return c;
}

matrix* mult(matrix* a, matrix* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (a->col != b->row) {
		fprintf(stderr, "Number of columns of first matrix does not match number of rows of second matrix\n");
		exit(1);
	}
	#endif
	int row = a->row;
	int col = b->col;
	int n = a->col;
	matrix* c = init_mat(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// c[i][j] = dot product of ith row of a and jth col of b (i.e sum over k of (a[i][k] * b[k][j]) )
			number sum = 0;
			for (int k = 0; k < n; k++) {
				sum += get(a, i, k) * get(b, k, j);
			}
			set(c, i, j, sum);
		}
	}
	return c;
}

vector* mult(matrix* a, vector* b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (a->col != b->size) {
		fprintf(stderr, "Number of columns of matrix does not match size of vector\n");
		exit(1);
	}
	#endif
	int inner_size = b->size;
	int size = a->row;
	vector* v = init_vec(size);
	for (int i = 0; i < size; i++) {
		// v[i] = sum over j of (a[i][j] * b[j])
		number sum = 0;
		for (int j = 0; j < inner_size; j++) {
			sum += get(a, i, j) * get(b, j);
		}
		set(v, i, sum);
	}

	return v;
}
