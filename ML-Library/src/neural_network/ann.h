#include "../mllib.h"
#include "../math/matrix.h"
#include "../processing/batch.h"
#include "layers.h"

#ifndef MLLIB_ANN_H
#define MLLIB_ANN_H

/*
struct ann_ {	
	matrix** weights;
	vector** biases;

	size_t* layers;
	size_t number_of_layers;
	number gamma;
	boolean is_classifier;
};
typedef struct ann_ ann;
*/

struct ann_ {
	ann_layer* layers;
	size_t number_of_layers;
	number gamma;
};
typedef struct ann_ NeuralNet;

/**
 * Refactoring code
 */
NeuralNet* initialize_nn(const size_t* dims, size_t n_layers, number gamma);
void free_nn(NeuralNet* nn);

/**
 * Training and testing of the neural network
 */
void train(NeuralNet* neural_network, m_batch* training_input, m_batch* training_output, size_t number_of_loops);
batch* pass_forward(NeuralNet* neural_network, batch* inputs);

#ifdef ML_LIB_DEBUG_MODE
void print_network(NeuralNet* neural_network);
#endif

#endif