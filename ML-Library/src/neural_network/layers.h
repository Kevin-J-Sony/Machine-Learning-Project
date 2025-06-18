#include "../mllib.h"
#include "../math/matrix.h"
#include "../processing/batch.h"

#ifndef MLLIB_LAYERS_H
#define MLLIB_LAYERS_H

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
};

layer_type i = DENSE;

#endif