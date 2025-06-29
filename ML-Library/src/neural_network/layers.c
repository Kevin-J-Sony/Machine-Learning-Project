#include "layers.h"

/**
 * Leaky ReLU for nonlinear transformation applied on matrix. Applied to all entries
 */
void leaky_relu(matrix* output, matrix* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->number_of_cols != input->number_of_cols) || (output->number_of_rows != input->number_of_rows)) {
		fprintf(stderr, "ERROR IN LEAKY RELU: output batch does not match input batch");
		exit(EXIT_FAILURE);
	}
	#endif

	number a = 0.001;
	for (int i = 0; i < input->number_of_rows; i++) {
		for (int j = 0; j < input->number_of_cols; j++) {
			number entry = input->m[i * input->number_of_cols + j];
			output->m[i * output->number_of_cols + j] = (entry > 0) ? entry : (a * entry);
		}
	}
}

/**
 * Derivative of Leaky ReLU applied on matrix. Applied to all entries
 */
void leaky_relu_derivative(matrix* output, matrix* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->number_of_cols != input->number_of_cols) || (output->number_of_rows != input->number_of_rows)) {
		fprintf(stderr, "ERROR IN LEAKY RELU DERIVATIVE: output batch does not match input batch.\n");
		exit(EXIT_FAILURE);
	}
	#endif
	
	number a = 0.001;
	for (int i = 0; i < input->number_of_rows; i++) {
		for (int j = 0; j < input->number_of_cols; j++) {
			number entry = input->m[i * input->number_of_cols + j];
			output->m[i * output->number_of_cols + j] = (entry > 0) ? 1.0 : a;
		}
	}
}

void sigmoid(matrix* output, matrix* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->number_of_cols != input->number_of_cols) || (output->number_of_rows != input->number_of_rows)) {
		fprintf(stderr, "ERROR IN LEAKY RELU DERIVATIVE: output batch does not match input batch.\n");
		exit(EXIT_FAILURE);
	}
	#endif
	
	for (int i = 0; i < input->number_of_rows; i++) {
		for (int j = 0; j < input->number_of_cols; j++) {
			number entry = VALUE_AT(input, i, j);
			entry = 1 / (1 + expf(-entry));
			VALUE_AT(output, i, j) = entry;
		}
	}

}

/**
 * DO NOT USE THIS METHOD UNLESS ABSOLUTELY NECESSARY
 */
void sigmoid_derivative(matrix* output, matrix* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->number_of_cols != input->number_of_cols) || (output->number_of_rows != input->number_of_rows)) {
		fprintf(stderr, "ERROR IN LEAKY RELU DERIVATIVE: output batch does not match input batch.\n");
		exit(EXIT_FAILURE);
	}
	#endif
	
	for (int i = 0; i < input->number_of_rows; i++) {
		for (int j = 0; j < input->number_of_cols; j++) {
			number entry = VALUE_AT(input, i, j);
			entry = 1 / (1 + expf(-entry));
			entry = entry * (1 - entry);
			VALUE_AT(output, i, j) = entry;
		}
	}
}


