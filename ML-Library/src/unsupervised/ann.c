#include "ann.h"

ann* initialize_ann(size_t* sizes, size_t number_of_layers) {
	ann* neural_network;

	#ifdef ML_LIB_DEBUG_MODE
	neural_network = (ann *)calloc(1, sizeof(ann));
	neural_network->layers = (size_t *)calloc(number_of_layers, sizeof(size_t));

	neural_network->biases = (vector **)calloc(number_of_layers - 1, sizeof(vector *));
	neural_network->weights = (matrix **)calloc(number_of_layers - 1, sizeof(matrix *));
	#else
	neural_network = (ann *)malloc(1, sizeof(ann));
	neural_network->layers = (size_t *)malloc(number_of_layers * sizeof(size_t));

	neural_network->biases = (vector **)malloc((number_of_layers - 1) * sizeof(vector *));
	neural_network->weights = (matrix **)malloc((number_of_layers - 1) * sizeof(matrix *));
	#endif

	for (int i = 0; i < number_of_layers - 1; i++) {
		neural_network->weights[i] = init_mat(sizes[i + 1], sizes[i]);
		neural_network->biases[i] = init_vec(sizes[i + 1]);

		neural_network->layers[i] = sizes[i];
	}
	neural_network->layers[number_of_layers - 1] = sizes[number_of_layers - 1];
	neural_network->number_of_layers = number_of_layers;

	return neural_network;
}

void deallocate_ann(ann* neural_network) {
	for (int i = 0; i < neural_network->number_of_layers - 1; i++) {
		del_mat(neural_network->weights[i]);
		del_vec(neural_network->biases[i]);
	}
	free(neural_network->weights);
	free(neural_network->biases);
	free(neural_network->layers);
}

void train(ann* neural_network, batch* training_input, batch* training_output) {
	#ifdef ML_LIB_DEBUG_MODE
	if (training_input->number_of_vectors != training_output->number_of_vectors) {
		fprintf(stdout, "ANN TRAINING ERROR: Number of inputs does not match number of outputs\n");
		exit(EXIT_FAILURE);
	}
	if (training_input->vector_size != neural_network->layers[0]) {
		fprintf(stdout, "ANN TRAINING ERROR: Size of inputs do not match input layer of neural network\n");
		exit(EXIT_FAILURE);
	}
	if (training_output->vector_size != neural_network->layers[neural_network->number_of_layers - 1]) {
		fprintf(stdout, "ANN TRAINING ERROR: Size of outputs does not match output layer of neural network\n");
		exit(EXIT_FAILURE);
	}
	#endif

	batch** intermediate_outputs;
	size_t io_batch_size = training_input->batch_size;
	size_t io_number_of_vectors = training_input->number_of_vectors;

	#ifdef ML_LIB_DEBUG_MODE
	intermediate_outputs = (batch **)calloc(neural_network->number_of_layers, sizeof(batch *));
	#else
	intermediate_outputs = (batch **)malloc(neural_network->number_of_layers * sizeof(batch *));
	#endif


	for (int i = 0; i < neural_network->number_of_layers; i++) {
		intermediate_outputs[i] = create_empty_batch(io_number_of_vectors, io_batch_size, neural_network->layers[i]);
	}
}

