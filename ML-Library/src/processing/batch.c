#include "batch.h"

/**
 * Create a structure over the inputs (represented as vectors). This is done to improve computing speeds.
 * More concretely, we could feed the neural network one input at a time to train it, but such methods is highly
 * inoptimal. If we have 1000 inputs for example, training one input at a time would mean the neural network would have to
 * iterate 1000 times. If instead, the neural network can accept 20 inputs at a time, it would only have to iterate
 * 50 times. It's a massive reduction, but requires additional scaffolding.
 * 
 * Of course, the benefits are only reaped when CUDA is employed.
 */
batch* create_batch(vector** huge_number_of_inputs, size_t number_of_inputs, size_t batch_size) {	
	
	size_t numb_of_batches = number_of_inputs / batch_size;
	numb_of_batches += (number_of_inputs % batch_size == 0) ? 1 : 0;
	
	batch* new_batch;

	#ifdef ML_LIB_DEBUG_MODE
	new_batch = (batch*)calloc(1, sizeof(batch));
	#else
	new_batch = (batch*)malloc(sizeof(batch));	
	#endif

	new_batch->number_of_batches = numb_of_batches;
	#ifdef ML_LIB_DEBUG_MODE
	new_batch->batches = (s_batch*)calloc(numb_of_batches, sizeof(s_batch));
	#else
	new_batch->batches = (s_batch*)malloc(numb_of_batches * sizeof(s_batch));
	#endif

	size_t inputs_remaining = number_of_inputs;
	int idx = 0;
	while (inputs_remaining > batch_size) {
		(new_batch->batches[idx]).number_of_inputs = batch_size;
		for (int jidx = 0; jidx < batch_size; jidx++) {
			(new_batch->batches[idx]).inputs[jidx] = huge_number_of_inputs[numb_of_batches * idx + jidx];
		}

		inputs_remaining -= batch_size;
		idx++;
	}

	#ifdef ML_LIB_DEBUG_MODE
	assert((idx == numb_of_batches - 1) && "Index is not as expected");
	#endif

	(new_batch->batches[idx]).number_of_inputs = inputs_remaining;
	for (int jidx = 0; jidx < inputs_remaining; jidx++) {
		(new_batch->batches[idx]).inputs[jidx] = huge_number_of_inputs[numb_of_batches * idx + jidx];
	}

	return new_batch;
}

/**
 * This function deletes the batch structure. The contents are not freed however.
 */
void delete_batch(batch* batch_to_delete) {
	free(batch_to_delete->batches);
	free(batch_to_delete);
}
