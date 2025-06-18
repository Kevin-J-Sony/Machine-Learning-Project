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
	number gamma;
	boolean is_classifier;
};
typedef struct ann_ ann;


ann* initialize_ann(size_t* sizes, size_t number_of_layers, boolean classification_task);
void deallocate_ann(ann* neural_network);

/**
 * Nonlinear functions and derivatives
 */
void nonlinear_transform_mat(matrix* output, matrix* input);
void nonlinear_transform_derivative_mat(matrix* output, matrix* input);

/**
 * Training and testing of the neural network
 */
void train(ann* neural_network, m_batch* training_input, m_batch* training_output, size_t number_of_loops);
batch* pass_forward(ann* neural_network, batch* inputs);

#ifdef ML_LIB_DEBUG_MODE
void print_network(ann* neural_network);
#endif

#endif