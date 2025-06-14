#include "batch.h"


/**
 * The batch will represent each input as a column, meaning
 */
batch* create_empty_batch(size_t number_of_vectors, size_t vec_size) {	
	batch* empty_batch;

	#ifdef ML_LIB_DEBUG_MODE
	empty_batch = (batch *)calloc(1, sizeof(batch));
	#else
	empty_batch = (batch *)malloc(sizeof(batch));
	#endif
	
	empty_batch->vector_size = vec_size;
	empty_batch->number_of_vectors = number_of_vectors;
	
	empty_batch->data = init_mat(vec_size, number_of_vectors);

	return empty_batch;
}

/**
 * This function deletes the batch structure. The contents are not freed however.
 */
void delete_batch(batch* batch_to_delete) {
	del_mat(batch_to_delete->data);
	free(batch_to_delete);
}

void load_data_into_batch(batch* empty_batch, vector** huge_number_of_data, size_t number_of_data) {
	size_t vector_size = huge_number_of_data[0]->size;

	#ifdef ML_LIB_DEBUG_MODE
	for (int i = 1; i < number_of_data; i++) {
		if (huge_number_of_data[i]->size != vector_size) {
			fprintf(stderr, "ERROR IN LOAD BATCH: Input sizes aren't consistent\n");
			exit(EXIT_FAILURE);
		}
	}

	if (empty_batch->vector_size != vector_size) {
		fprintf(stderr, "ERROR IN LOAD BATCH: The empty batch vector size is not consistent\n");
		exit(EXIT_FAILURE);
	}

	if (empty_batch->number_of_vectors != number_of_data) {
		fprintf(stderr, "ERROR IN LOAD BATCH: The number of inputs the empty batch vector can accept is not equal to the number of data vectors supplied.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int data_idx = 0; data_idx < number_of_data; data_idx++) {
		for (int entry = 0; entry < vector_size; entry++) {
			empty_batch->data->m[entry * number_of_data + data_idx] = huge_number_of_data[data_idx]->v[entry];
		}
	}
}


m_batch* load_data_into_batches(vector** huge_number_of_data, size_t number_of_data, size_t batch_size) {
	#ifdef ML_LIB_DEBUG_MODE
	for (int i = 0; i < number_of_data; i++) {
		if (huge_number_of_data[i]->size != huge_number_of_data[0]->size) {
			fprintf(stderr, "ERROR IN LOAD MANY BATCHES: The sizes of the vectors are inconsistent");
			exit(EXIT_FAILURE);
		}
	}
	#endif

	m_batch* many_batches;
	#ifdef ML_LIB_DEBUG_MODE
	many_batches = (m_batch *)calloc(1, sizeof(m_batch));
	#else
	many_batches = (m_batch *)malloc(sizeof(m_batch));
	#endif

	size_t number_of_batches = number_of_data / batch_size;
	many_batches->number_of_batches = number_of_batches;
	many_batches->total_number_of_vectors = number_of_data;
	
	many_batches->vector_size = huge_number_of_data[0]->size; // should be same across all values

	#ifdef ML_LIB_DEBUG_MODE
	many_batches->ray_of_batches = (batch **)calloc(number_of_batches, sizeof(batch *));
	#else
	many_batches->ray_of_batches = (batch **)malloc(number_of_batches * sizeof(batch *));
	#endif

	for (int i = 0; i < number_of_batches; i++) {
		many_batches->ray_of_batches[i] = create_empty_batch(batch_size, many_batches->vector_size);

		for (int j = 0; j < many_batches->vector_size; j++) {
			for (int k = 0; k < batch_size; k++) {
				VALUE_AT(many_batches->ray_of_batches[i]->data, j, k) = huge_number_of_data[i * batch_size + k]->v[j];
			}
		}
	}
	
	return many_batches;
}


void delete_batches(m_batch* many_batches) {
	for (int i = 0; i < many_batches->number_of_batches; i++) {
		delete_batch(many_batches->ray_of_batches[i]);
	}
	free(many_batches->ray_of_batches);
	free(many_batches);
}



/**
 * Each column in input_batch_vectors is an input to be multiplied by mat, and the output goes to output_batch_vectors.
 */
void multiply_batch_by_matrix(batch *output_batch_vectors, matrix *mat, batch *input_batch_vectors) {
	#ifdef ML_LIB_DEBUG_MODE
	if (input_batch_vectors->vector_size != mat->number_of_cols) {
		fprintf(stderr, "ERROR IN MULTIPLY BATCH BY MATRIX: The inputs of the batch does not match the number of columns in the matrix.\n");
		exit(EXIT_FAILURE);
	}
	
	if (output_batch_vectors->vector_size != mat->number_of_rows) {
		fprintf(stderr, "ERROR IN MULTIPLY BATCH BY MATRIX: The ouputs of the batch does not match the number of rows in the matrix.\n");
		exit(EXIT_FAILURE);
	}
	
	if (output_batch_vectors->number_of_vectors != input_batch_vectors->number_of_vectors) {
		fprintf(stderr, "ERROR IN MULTIPLY BATCH BY MATRIX: The number of vectors allocated for the output does not match the number of vectors from the input\n");
		exit(EXIT_FAILURE);
	}
	#endif

	matrix_mult(output_batch_vectors->data, mat, input_batch_vectors->data);
}

/**
 * Add a vector to each column in the input batch and store it in the output batch.
 */
void add_vector_to_batch(batch* output_batch, batch* input_batch, vector* vec) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (input_batch->vector_size != output_batch->vector_size) || 
		 (input_batch->number_of_vectors != output_batch->number_of_vectors) ) {
		fprintf(stderr, "ERROR IN ADD VECTOR TO BATCH: The dimensions of the input batch and output batch do not match.\n");
		exit(EXIT_FAILURE);
	}

	if (input_batch->vector_size != vec->size) {
		fprintf(stderr, "ERROR IN ADD VECTOR TO BATCH: The size of the vector does not match the column lengths of the batches.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	add_vector_to_matrix(output_batch->data, input_batch->data, vec);
}

/**
 * Hadamard product is simply entrywise product of matrices. Very intuitive to understand
 */
void batch_hadamard_product(batch* output, batch* product_one, batch* product_two) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (product_one->vector_size != product_two->vector_size) ||
		 (product_one->number_of_vectors != product_two->number_of_vectors)) {
		fprintf(stderr, "ERROR IN BATCH HADAMARD PRODUCT: Dimensions of inputs do not match\n");
		exit(EXIT_FAILURE);
	}

	if ( (product_one->vector_size != output->vector_size) ||
		 (product_one->number_of_vectors != output->number_of_vectors)) {
		fprintf(stderr, "ERROR IN BATCH HADAMARD PRODUCT: Dimensions of output does not match dimensions of inputs\n");
		exit(EXIT_FAILURE);
	}
	#endif

	matrix_entrywise_product(output->data, product_one->data, product_two->data);
}