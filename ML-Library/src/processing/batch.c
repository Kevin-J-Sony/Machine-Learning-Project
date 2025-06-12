#include "batch.h"


batch* create_empty_batch(size_t number_of_vectors, size_t b_size, size_t vec_size) {	
	batch* empty_batch;

	#ifdef ML_LIB_DEBUG_MODE
	size_t vector_size = huge_number_of_data[0]->size;
	for (int i = 1; i < number_of_data; i++) {
		if (huge_number_of_data[i]->size != vector_size) {
			fprintf(stderr, "ERROR IN CREATE BATCH: Input sizes aren't consistent\n");
			exit(EXIT_FAILURE);
		}
	}
	#endif

	return (batch *)0;
}

/**
 * This function deletes the batch structure. The contents are not freed however.
 */
void delete_batch(batch* batch_to_delete) {
	free(batch_to_delete->data);
	free(batch_to_delete);
}

void load_data_into_batch(batch* empty_batch, vector** huge_number_of_data, size_t number_of_data, size_t batch_size) {

	#ifdef ML_LIB_DEBUG_MODE
	size_t vector_size = huge_number_of_data[0]->size;
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
	#endif

	size_t numb_of_batches = number_of_data / batch_size;
	numb_of_batches += (number_of_data % batch_size == 0) ? 0 : 1;
	
	empty_batch->number_of_vectors = huge_number_of_data[0]->size;

	#ifdef ML_LIB_DEBUG_MODE
	empty_batch->batches = (s_batch*)calloc(numb_of_batches, sizeof(s_batch));
	#else
	empty_batch->batches = (s_batch*)malloc(numb_of_batches * sizeof(s_batch));
	#endif

	size_t inputs_remaining = number_of_inputs;
	int idx = 0;
	while (inputs_remaining > batch_size) {
		(empty_batch->batches[idx]).number_of_inputs = batch_size;
		(empty_batch->batches[idx]).input_size = empty_batch->input_size;
		
		#ifdef ML_LIB_DEBUG_MODE
		(empty_batch->batches[idx]).inputs = (vector **)calloc(batch_size, sizeof(vector *));
		#else
		(empty_batch->batches[idx]).inputs = (vector **)malloc(batch_size * sizeof(vector *));
		#endif
		
		for (int jidx = 0; jidx < batch_size; jidx++) {
			(empty_batch->batches[idx]).inputs[jidx] = huge_number_of_inputs[numb_of_batches * idx + jidx];
		}

		inputs_remaining -= batch_size;
		idx++;
	}

	#ifdef ML_LIB_DEBUG_MODE
	if (idx != numb_of_batches - 1) {
		fprintf(stderr, "LOGICAL ERROR IN CREATE BATCH: Index is not as expected\n");
		exit(EXIT_FAILURE);
	}
	#endif

	(empty_batch->batches[idx]).number_of_inputs = inputs_remaining;
	(empty_batch->batches[idx]).input_size = empty_batch->input_size;

	#ifdef ML_LIB_DEBUG_MODE
	(empty_batch->batches[idx]).inputs = (vector **)calloc(inputs_remaining, sizeof(vector *));
	#else
	(empty_batch->batches[idx]).inputs = (vector **)malloc(inputs_remaining * sizeof(vector *));
	#endif

	for (int jidx = 0; jidx < inputs_remaining; jidx++) {
		(empty_batch->batches[idx]).inputs[jidx] = huge_number_of_inputs[numb_of_batches * idx + jidx];
	}
}



/**
 * Multiply each vector in batch_of_vectors by mat
 */
void multiply_single_batch(batch *output_sbatch_of_vectors, matrix *mat, batch *sbatch_of_vectors) {
	#ifdef ML_LIB_DEBUG_MODE
	if (sbatch_of_vectors->input_size != mat->number_of_cols) {
		fprintf(stderr, "ERROR IN MULTIPLY SINGULAR BATCH: The inputs of the batch does not match the number of columns in the matrix.\n");
		exit(EXIT_FAILURE);
	} else if (output_sbatch_of_vectors->input_size != mat->number_of_rows) {
		fprintf(stderr, "ERROR IN MULTIPLY SINGULAR BATCH: The ouputs of the batch does not match the number of rows in the matrix.\n");
		exit(EXIT_FAILURE);
	} else if (output_sbatch_of_vectors->number_of_inputs != sbatch_of_vectors->number_of_inputs) {
		fprintf(stderr, "ERROR IN MULTIPLY SINGULAR BATCH: The number of vectors allocated for the output does not match the number of vectors from the input\n");
		exit(EXIT_FAILURE);
	}
	#endif

	vector** outputs = output_sbatch_of_vectors->inputs;
	vector** inputs = sbatch_of_vectors->inputs;
	for (int i = 0; i < output_sbatch_of_vectors->number_of_inputs; i++) {
		matrix_vector_mult(outputs[i], mat, inputs[i]);
	}
}
