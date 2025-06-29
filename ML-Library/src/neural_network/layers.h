#include "../mllib.h"
#include "../math/matrix.h"
#include "../processing/batch.h"

#ifndef MLLIB_LAYERS_H
#define MLLIB_LAYERS_H

typedef void (*activation_function)(matrix* out, matrix* in);

struct ann_layer_ {
	matrix* W;
	vector* b;
	activation_function a;
	activation_function a_prime;

	size_t dim_input;
	size_t dim_output;
};
typedef struct ann_layer_ ann_layer;

/**
 * Nonlinear activation functions and their derivatives
 */
void leaky_relu(matrix* output, matrix* input);
void leaky_relu_derivative(matrix* output, matrix* input);


void sigmoid(matrix* output, matrix* input);
void sigmoid_derivative(matrix* output, matrix* input);



#endif