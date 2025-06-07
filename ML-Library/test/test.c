/**
 * 		Testing various methods in ML-Library
 */
#include "../src/math/matrix.h"
#include <stdio.h>

void print_mat(matrix* mat) {
	for (int i = 0; i < mat->row; i++) {
		for (int j = 0; j < mat->col; j++) {
			printf("%d ", get_mat(m, i, j));
		}
		printf("\n");
	}
}

void print_vec(vector* vec) {
	for (int i = 0; i < vec->size; i++) {
		printf("%d\n", get_vec(vec, i));
	}
}

int main() {
	matrix* mat1 = init_mat(3, 3);
	matrix* mat2 = init_mat(3, 3);

	print_mat(mat1);

	return 0;
}