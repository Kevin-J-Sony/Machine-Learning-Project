/**
 * 		Testing various methods in ML-Library
 */
#include "../src/math/matrix.h"

void print_mat(matrix* mat) {
	size_t nrows = mat->number_of_rows;
	size_t ncols = mat->number_of_cols;
	number** m = mat->m;

	fprintf(stdout, "Matrix info\n----------\n");
	fprintf(stdout, "Number of rows: %d \t Number of columns: %d\n", nrows, ncols);

	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			fprintf(stdout, "%f ", (mat->m[i])[j]);
		}
		fprintf(stdout, "\n");
	}
}

void print_vec(vector* vec) {
	size_t size = vec->size;
	number* v = vec->v;

	fprintf(stdout, "Vector info\n----------\n");
	fprintf(stdout, "Size of vector: %d \n", size);

	for (int i = 0; i < size; i++) {
		fprintf(stdout, "%f ", v[i]);
	}
	fprintf(stdout, "\n");
}

void test_mat_add() {
	matrix* mat1 = init_mat(4, 4);
	matrix* mat2 = init_mat(4, 4);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mat1->m[i][j] = rand() % 20 + 1;
			mat2->m[j][i] = rand() % 20 + 1;
		}
	}

	matrix* mat3 = init_mat(4, 4);
	matrix_add(mat3, mat1, mat2);
	print_mat(mat1);
	print_mat(mat2);
	print_mat(mat3);
}

void test_mat_mult() {
	matrix* mat1 = init_mat(3, 4);
	matrix* mat2 = init_mat(4, 3);

	int f = 1;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			mat1->m[i][j] = f;
			mat2->m[j][i] = f;			
			f++;
		}
	}

	matrix* mat3 = init_mat(mat1->number_of_rows, mat2->number_of_cols);
	matrix_mult(mat3, mat1, mat2);
	print_mat(mat1);
	print_mat(mat2);
	print_mat(mat3);
}

void test_mat_vec_mult() {
	matrix* mat = init_mat(3, 4);
	vector* vec = init_vec(4);

	int f = 1;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			mat->m[i][j] = f;
			f++;
			vec->v[j] = 1;
		}
	}

	vector* out = init_vec(mat->number_of_rows);
	matrix_vector_mult(out, mat, vec);
	print_vec(vec);
	print_mat(mat);
	print_vec(out);
}

int main() {
	srand(10);

	fprintf(stdout, "\n\nBEGIN TESTING\n\n");



	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF MATRIX ADDITION\n--------------------\n");
	test_mat_add();
	fprintf(stdout, "\n--------------------\nEND TESTING OF MATRIX ADDITION\n--------------------\n");

	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF MATRIX MULTIPLICATION\n--------------------\n");
	test_mat_mult();
	fprintf(stdout, "\n--------------------\nEND TESTING OF MATRIX MULTIPLICATION\n--------------------\n");

	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF MATRIX-VECTOR MULTIPLICATION\n--------------------\n");
	test_mat_vec_mult();
	fprintf(stdout, "\n--------------------\nEND TESTING OF MATRIX-VECTOR MULTIPLICATION\n--------------------\n");

	fprintf(stdout, "\n\nEND TESTING\n\n");
	return 0;
}