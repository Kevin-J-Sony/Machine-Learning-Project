#include "../mllib.h"
#include "../math/matrix.h"
#include "../processing/batch.h"

#ifndef MLLIB_LAYERS_H
#define MLLIB_LAYERS_H

typedef void (*activation_function)(matrix* out, matrix* in);

enum layer_type_ {
	DENSE,
	CONVOLUTION,
	DROPOUT,
	CLASSIFICATION_OUTPUT,
	REGRESSION_OUTPUT
};
typedef enum layer_type_ layer_type;

struct ann_layer_ {
	layer_type type;

	matrix* weight;
	vector* bias;
	activation_function a;
};


/**
 * Nonlinear activation functions and their derivatives
 */
void leaky_relu(matrix* output, matrix* input);
void leaky_relu_derivative(matrix* output, matrix* input);


#endif