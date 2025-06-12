#include "../mllib.h"
#include "../math/matrix.h"
#include "../processing/batch.h"

#ifndef MLLIB_ANN_H
#define MLLIB_ANN_H

struct ann_ {
	
	/**
	 * This structure contains an array of pointers to matrices and vectors, due to the way the matrix
	 * and vector initialization is set up.
	 */
	matrix** weights;
	vector** biases;
	size_t* layers;
	size_t number_of_layers;
};
typedef struct ann_ ann;


ann* initialize_ann(size_t* sizes, size_t number_of_layers);
void deallocate_ann(ann* neural_network);

/**
 * For each s_batch, we pass in all the s_batches at the same time, resulting in 
 */
void train(ann* neural_network, vector** training_input, vector** training_output, size_t batch_size);


#endif