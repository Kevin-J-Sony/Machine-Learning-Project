#include "../mllib.h"
#include "../math/matrix.h"

#ifndef MLLIB_BATCH_H
#define MLLIB_BATCH_H

struct singular_batch_ {
	vector* inputs;
	size_t number_of_inputs;
};
typedef struct singular_batch_ s_batch;

struct batch_ {
	s_batch* batches;
	size_t number_of_batches;
};
typedef struct batch_ batch;


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
	while (inputs_remaining > batch_size) {
		batch_size[];
		inputs_remaining -= batch_size;
	}
}

#endif