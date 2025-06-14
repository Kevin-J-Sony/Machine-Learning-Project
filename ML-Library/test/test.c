/**
 * 		Testing various methods in ML-Library
 */
#include "../src/math/matrix.h"
#include "../src/processing/batch.h"
#include "../src/unsupervised/ann.h"

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

void print_batch(batch* bat) {
	size_t size = bat->number_of_vectors;
	matrix* data = bat->data;

	fprintf(stdout, "----------\nBatch info\n");
	fprintf(stdout, "Number of vectors: %lu \n", size);

	for (int i = 0; i < size; i++) {		
		fprintf(stdout, "Vector %d: ", i+1);
		for (int j = 0; j < bat->vector_size; j++) {
			fprintf(stdout, "%f ", data->m[j * size + i]);
		}
		fprintf(stdout, "\n");
	}
}

void print_network(ann* neural_network) {
	fprintf(stdout, "----------\nNeural network info\n");

	size_t number_of_layers = neural_network->number_of_layers;
	for (int i = 0; i < number_of_layers - 1; i++) {
		print_mat(neural_network->weights[i]);
		print_vec(neural_network->biases[i]);
	}
}








void test_mat_add() {
	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF MATRIX ADDITION\n--------------------\n");

	size_t x = 4;
	size_t y = 4;

	matrix* mat1 = init_mat(x, y);
	matrix* mat2 = init_mat(x, y);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mat1->m[i * y + j] = rand() % 20 + 1;
			mat2->m[j * x + i] = rand() % 20 + 1;
		}
	}

	matrix* mat3 = init_mat(x, y);
	matrix_add(mat3, mat1, mat2);
	print_mat(mat1);
	print_mat(mat2);
	print_mat(mat3);

	del_mat(mat1);
	del_mat(mat2);
	del_mat(mat3);

	fprintf(stdout, "\n--------------------\nEND TESTING OF MATRIX ADDITION\n--------------------\n");

}

void test_mat_mult() {
	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF MATRIX MULTIPLICATION\n--------------------\n");

	size_t x = 3;
	size_t y = 4;

	matrix* mat1 = init_mat(x, y);
	matrix* mat2 = init_mat(y, x);

	int f = 1;
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			mat1->m[i * y + j] = f;
			mat2->m[j * x + i] = f;
			f++;
		}
	}

	matrix* mat3 = init_mat(x, x);
	matrix_mult(mat3, mat1, mat2);
	print_mat(mat1);
	print_mat(mat2);
	print_mat(mat3);

	del_mat(mat1);
	del_mat(mat2);
	del_mat(mat3);

	fprintf(stdout, "\n--------------------\nEND TESTING OF MATRIX MULTIPLICATION\n--------------------\n");
}

void test_mat_vec_mult() {
	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF MATRIX-VECTOR MULTIPLICATION\n--------------------\n");

	matrix* mat = init_mat(3, 4);
	vector* vec = init_vec(4);

	int f = 1;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			mat->m[i * 4 + j] = f;
			f++;
			vec->v[j] = 1;
		}
	}

	vector* out = init_vec(mat->number_of_rows);
	matrix_vector_mult(out, mat, vec);
	print_vec(vec);
	print_mat(mat);
	print_vec(out);

	del_vec(vec);
	del_mat(mat);
	del_vec(out);

	fprintf(stdout, "\n--------------------\nEND TESTING OF MATRIX-VECTOR MULTIPLICATION\n--------------------\n");

}

void test_batch() {
	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF BATCH OPERATIONS\n--------------------\n");

	size_t number_of_inputs = 4;
	size_t size_of_inputs = 4;
	vector** inputs = (vector **)malloc(number_of_inputs * sizeof(vector *));
	number tally = 1;
	for (int i = 0; i < number_of_inputs; i++) {
		inputs[i] = init_vec(size_of_inputs);
		for (int j = 0; j < size_of_inputs; j++) {
			(inputs[i]->v)[j] = tally;
		}
	}

	batch* input_batch = create_empty_batch(number_of_inputs, size_of_inputs);
	load_data_into_batch(input_batch, inputs, number_of_inputs);
	print_batch(input_batch);

	size_t size_of_outputs = 5;
	matrix* mat = init_mat(size_of_outputs, size_of_inputs);
	tally = 1;
	for (int i = 0; i < size_of_outputs; i++) {
		for (int j = 0; j < size_of_inputs; j++) {
			mat->m[i * size_of_inputs + j] = tally++;
		}
	}

	print_mat(mat);

	batch* output_batch = create_empty_batch(number_of_inputs, size_of_outputs);
	print_batch(output_batch);

	del_mat(mat);
	delete_batch(input_batch);
	delete_batch(output_batch);

	for (int i = 0; i < number_of_inputs; i++) {
		del_vec(inputs[i]);
	}
	free(inputs);

	fprintf(stdout, "\n--------------------\nEND TESTING OF BATCH OPERATIONS\n--------------------\n");
}


void test_ann() {
	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF NEURAL NETWORK INITIALIZATION AND PASS THROUGH\n--------------------\n");

	size_t number_of_inputs = 4;
	size_t size_of_inputs = 4;
	vector** inputs = (vector **)malloc(number_of_inputs * sizeof(vector *));
	number tally = 1;
	for (int i = 0; i < number_of_inputs; i++) {
		inputs[i] = init_vec(size_of_inputs);
		for (int j = 0; j < size_of_inputs; j++) {
			(inputs[i]->v)[j] = tally;
		}
	}

	batch* input_batch = create_empty_batch(number_of_inputs, size_of_inputs);
	load_data_into_batch(input_batch, inputs, number_of_inputs);
	print_batch(input_batch);

	size_t size_of_outputs = 5;
	matrix* mat = init_mat(size_of_outputs, size_of_inputs);
	tally = 1;
	for (int i = 0; i < size_of_outputs; i++) {
		for (int j = 0; j < size_of_inputs; j++) {
			mat->m[i * size_of_inputs + j] = tally++;
		}
	}

	print_mat(mat);

	batch* output_batch = create_empty_batch(number_of_inputs, size_of_outputs);
	print_batch(output_batch);

	del_mat(mat);
	delete_batch(input_batch);
	delete_batch(output_batch);

	for (int i = 0; i < number_of_inputs; i++) {
		del_vec(inputs[i]);
	}
	free(inputs);

	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF NEURAL NETWORK INITIALIZATION AND PASS THROUGH\n--------------------\n");
}


int main() {
	srand(10);	// set the seed to reproduce results

	fprintf(stdout, "\n\nBEGIN TESTING\n\n");

	// test_mat_add();
	// test_mat_mult();
	// test_mat_vec_mult();
	// test_batch();

	size_t* sizes = (size_t *)calloc(3, sizeof(size_t));
	sizes[0] = 4; sizes[1] = 4; sizes[2] = 4;
	ann* nn = initialize_ann(sizes, 3);
	// print_network(nn);
	free(sizes);


	// Create some batches of inputs
	batch* batch_input = create_empty_batch(16, 4);
	batch* batch_output = create_empty_batch(16, 4);

	vector** data = (vector **) calloc(16, sizeof(vector *));
	
	for (int i = 0; i < 16; i++) {
		data[i] = init_vec(4);
		int t = i;
		for (int j = 0; j < 4; j++) {
			data[i]->v[j] = t % 2;
			t = t >> 1;
		}
	}

	load_data_into_batch(batch_input, data, 16);
	load_data_into_batch(batch_output, data, 16);

	train(nn, batch_input, batch_output);

	delete_batch(batch_input);
	delete_batch(batch_output);

	for (int i = 0; i < 16; i++) {
		del_vec(data[i]);
	}
	free(data);

	deallocate_ann(nn);
	fprintf(stdout, "\n\nEND TESTING\n\n");
	return 0;
}