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

void print_batch_compact_form(batch* bat) {
	size_t size = bat->number_of_vectors;
	matrix* data = bat->data;

	fprintf(stdout, "----------\nCompact Batch Representation\n");
	fprintf(stdout, "Dimension (%lu x %lu) \n", data->number_of_rows, data->number_of_cols);

	for (int i = 0; i < data->number_of_rows; i++) {		
		for (int j = 0; j < data->number_of_cols; j++) {
			fprintf(stdout, "%f ", data->m[i * data->number_of_cols + j]);
		}
		fprintf(stdout, "\n");
	}
}

void print_many_batches(m_batch* mbat) {
	size_t number_of_batches = mbat->number_of_batches;

	fprintf(stdout, "----------\nMultiple Batches Data Structure\n");
	
	for (int i = 0; i < number_of_batches; i++) {		
		fprintf(stdout, "Batch %d\n", i+1);
		batch* b = mbat->ray_of_batches[i];
		for (int j = 0; j < b->number_of_vectors; j++) {
			fprintf(stdout, "Vector %d: ", j+1);
			for (int k = 0; k < b->vector_size; k++) {
				fprintf(stdout, "%f ", VALUE_AT(b->data, k, j));
			}
			fprintf(stdout,"\n");
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

void test_batch_mat_mult() {
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

	size_t size_of_outputs = 5;
	matrix* mat = init_mat(size_of_outputs, size_of_inputs);
	tally = 1;
	for (int i = 0; i < size_of_outputs; i++) {
		for (int j = 0; j < size_of_inputs; j++) {
			mat->m[i * size_of_inputs + j] = tally++;
		}
	}


	batch* output_batch = create_empty_batch(number_of_inputs, size_of_outputs);
	matrix_mult(output_batch->data, mat, input_batch->data);

	print_mat(mat);
	print_batch_compact_form(input_batch);
	print_batch_compact_form(output_batch);

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

	size_t* sizes = (size_t *)calloc(3, sizeof(size_t));
	sizes[0] = 4; sizes[1] = 4; sizes[2] = 4;
	ann* nn = initialize_ann(sizes, 3);
	// print_network(nn);
	free(sizes);


	// Create some batches of inputs
	/*
	m_batch* mb_input = (m_batch *)calloc(1, sizeof(m_batch));
	mb_input->ray_of_batches = (batch **)calloc(1, sizeof(batch *));
	mb_input->ray_of_batches[0] = create_empty_batch(16, 4);
	mb_input->number_of_batches = 1;
	mb_input->total_number_of_vectors = 16;
	mb_input->vector_size = 4;

	m_batch* mb_output = (m_batch *)calloc(1, sizeof(m_batch));
	mb_output->ray_of_batches = (batch **)calloc(1, sizeof(batch *));
	mb_output->ray_of_batches[0] = create_empty_batch(16, 4);
	mb_output->number_of_batches = 1;
	mb_output->total_number_of_vectors = 16;
	mb_output->vector_size = 4;
	*/

	// batch* batch_input = create_empty_batch(16, 4);
	// batch* batch_output = create_empty_batch(16, 4);

	vector** data = (vector **) calloc(16, sizeof(vector *));
	
	for (int i = 0; i < 16; i++) {
		data[i] = init_vec(4);
		int t = i;
		for (int j = 0; j < 4; j++) {
			data[i]->v[j] = t % 2;
			t = t >> 1;
		}
	}

	// load_data_into_batch(mb_input->ray_of_batches[0], data, 16);
	// load_data_into_batch(mb_output->ray_of_batches[0], data, 16);

	m_batch* mb_input = load_data_into_batches(data, 16, 16);
	m_batch* mb_output = load_data_into_batches(data, 16, 16);
	print_many_batches(mb_input);
	print_many_batches(mb_output);

	train(nn, mb_input, mb_output);
	// print_network(nn);

	delete_batches(mb_input);
	delete_batches(mb_output);

	for (int i = 0; i < 16; i++) {
		del_vec(data[i]);
	}
	free(data);

	deallocate_ann(nn);

	fprintf(stdout, "\n--------------------\nBEGIN TESTING OF NEURAL NETWORK INITIALIZATION AND PASS THROUGH\n--------------------\n");
}


int main() {
	srand(10);	// set the seed to reproduce results

	fprintf(stdout, "\n\nBEGIN TESTING\n\n");

	// test_mat_add();
	// test_mat_mult();
	// test_mat_vec_mult();
	// test_batch();
	test_ann();

	fprintf(stdout, "\n\nEND TESTING\n\n");
	return 0;
}