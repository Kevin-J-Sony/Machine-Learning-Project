#include "../mllib.h"
#include "../math/matrix.h"

#ifndef MLLIB_BATCH_H
#define MLLIB_BATCH_H

/**
 * The batch contains an array of pointers to the inputs, not an array of the inputs itself
 * This is to make setting up much more easier.
 */
struct singular_batch_ {
	vector** inputs;
	size_t number_of_inputs;
};
typedef struct singular_batch_ s_batch;

struct batch_ {
	s_batch* batches;
	size_t number_of_batches;
};
typedef struct batch_ batch;

#endif