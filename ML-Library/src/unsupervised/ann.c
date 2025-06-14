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

		for (int j = 0; j < sizes[i + 1]; j++) {
			for (int k = 0; k < sizes[i]; k++) {
				VALUE_AT(neural_network->weights[i], j, k) = 1;
			}
			neural_network->biases[i]->v[j] = 1;
		}

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
	free(neural_network);
}




/**
 * Leaky ReLU for nonlinear transformation. Applied to all entries
 */
void nonlinear_transform(batch* output, batch* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->vector_size != input->vector_size) || (output->number_of_vectors != input->number_of_vectors)) {
		fprintf(stderr, "ERROR IN NONLINEAR TRANSFORM: output batch does not match input batch");
	}
	#endif

	for (int i = 0; i < input->vector_size; i++) {
		for (int j = 0; j < input->number_of_vectors; j++) {
			number entry = input->data->m[i * input->number_of_vectors + j];
			output->data->m[i * input->number_of_vectors + j] = (entry > 0) ? entry : (0.1 * entry);
		}
	}
}

/**
 * Leaky ReLU for nonlinear transformation applied on matrix instead of batch. Applied to all entries
 */
void nonlinear_transform_mat(matrix* output, matrix* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->number_of_cols != input->number_of_cols) || (output->number_of_rows != input->number_of_rows)) {
		fprintf(stderr, "ERROR IN NONLINEAR TRANSFORM: output batch does not match input batch");
	}
	#endif

	for (int i = 0; i < input->number_of_rows; i++) {
		for (int j = 0; j < input->number_of_cols; j++) {
			number entry = input->m[i * input->number_of_cols + j];
			output->m[i * output->number_of_cols + j] = (entry > 0) ? entry : (0.1 * entry);
		}
	}
}

/**
 * Derivative of Leaky ReLU. Applied to all entries
 */
void nonlinear_transform_derivative(batch* output, batch* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->vector_size != input->vector_size) || (output->number_of_vectors != input->number_of_vectors)) {
		fprintf(stderr, "ERROR IN NONLINEAR TRANSFORM DERIVATIVE: output batch does not match input batch");
	}
	#endif

	for (int i = 0; i < input->vector_size; i++) {
		for (int j = 0; j < input->number_of_vectors; j++) {
			number entry = input->data->m[i * input->number_of_vectors + j];
			output->data->m[i * input->number_of_vectors + j] = (entry > 0) ? 1.0 : 0.1;
		}
	}
}

/**
 * Derivative of Leaky ReLU applied on matrix instead of batch. Applied to all entries
 */
void nonlinear_transform_derivative_mat(matrix* output, matrix* input) {
	#ifdef ML_LIB_DEBUG_MODE
	if ( (output->number_of_cols != input->number_of_cols) || (output->number_of_rows != input->number_of_rows)) {
		fprintf(stderr, "ERROR IN NONLINEAR TRANSFORM: output batch does not match input batch");
	}
	#endif

	for (int i = 0; i < input->number_of_rows; i++) {
		for (int j = 0; j < input->number_of_cols; j++) {
			number entry = input->m[i * input->number_of_cols + j];
			output->m[i * output->number_of_cols + j] = (entry > 0) ? 1.0 : 0.1;
		}
	}
}



/**
 * Training function for the neural network. Accepts a batch of inputs and a batch of outputs.
 */
void train(ann* neural_network, m_batch* many_batches_training_input, m_batch* many_batches_training_output) {
	#ifdef ML_LIB_DEBUG_MODE
	if (many_batches_training_input->total_number_of_vectors != many_batches_training_output->total_number_of_vectors) {
		fprintf(stderr, "ANN TRAINING ERROR: Number of inputs does not match number of outputs\n");
		exit(EXIT_FAILURE);
	}
	if (many_batches_training_input->vector_size != neural_network->layers[0]) {
		fprintf(stderr, "ANN TRAINING ERROR: Size of inputs do not match input layer of neural network\n");
		exit(EXIT_FAILURE);
	}
	if (many_batches_training_output->vector_size != neural_network->layers[neural_network->number_of_layers - 1]) {
		fprintf(stderr, "ANN TRAINING ERROR: Size of outputs does not match output layer of neural network\n");
		exit(EXIT_FAILURE);
	}
	#endif

	// store weights and biases
	// matrix** weights = neural_network->weights;
	// vector** biases = neural_network->biases;
	size_t number_of_layers = neural_network->number_of_layers;


	int nloops = 2;
	int idx = 0;
	int curr_nloops = 0;
	while (curr_nloops < nloops * many_batches_training_input->number_of_batches) {
		batch* training_input = many_batches_training_input->ray_of_batches[idx % many_batches_training_input->number_of_batches];
		batch* training_output = many_batches_training_output->ray_of_batches[idx % many_batches_training_output->number_of_batches];
		idx = idx + 1;
		/*
		batch** linear_intermediate_outputs;
		batch** z_intermediate_outputs;
		batch** y_intermediate_outputs;
		*/


		matrix** linear_intermediate_outputs;
		matrix** z_intermediate_outputs;
		matrix** y_intermediate_outputs;

		size_t io_number_of_vectors = training_input->number_of_vectors;

		#ifdef ML_LIB_DEBUG_MODE
		/*
		linear_intermediate_outputs = (batch **)calloc(neural_network->number_of_layers, sizeof(batch *));
		y_intermediate_outputs = (batch **)calloc(neural_network->number_of_layers, sizeof(batch *));
		z_intermediate_outputs = (batch **)calloc(neural_network->number_of_layers, sizeof(batch *));
		*/

		linear_intermediate_outputs = (matrix **)calloc(neural_network->number_of_layers, sizeof(matrix *));
		y_intermediate_outputs = (matrix **)calloc(neural_network->number_of_layers, sizeof(matrix *));
		z_intermediate_outputs = (matrix **)calloc(neural_network->number_of_layers, sizeof(matrix *));
		#else
		/*
		linear_intermediate_outputs = (batch **)malloc(neural_network->number_of_layers * sizeof(batch *));
		y_intermediate_outputs = (batch **)malloc(neural_network->number_of_layers * sizeof(batch *));
		z_intermediate_outputs = (batch **)malloc(neural_network->number_of_layers * sizeof(batch *));
		*/

		linear_intermediate_outputs = (matrix **)malloc(neural_network->number_of_layers * sizeof(matrix *));
		y_intermediate_outputs = (matrix **)malloc(neural_network->number_of_layers * sizeof(matrix *));
		z_intermediate_outputs = (matrix **)malloc(neural_network->number_of_layers * sizeof(matrix *));
		#endif


		for (int i = 0; i < number_of_layers; i++) {
			/*
			linear_intermediate_outputs[i] = create_empty_batch(io_number_of_vectors, neural_network->layers[i]);
			z_intermediate_outputs[i] = create_empty_batch(io_number_of_vectors, neural_network->layers[i]);
			y_intermediate_outputs[i] = create_empty_batch(io_number_of_vectors, neural_network->layers[i]);
			*/
			linear_intermediate_outputs[i] = init_mat(neural_network->layers[i], io_number_of_vectors);
			z_intermediate_outputs[i] = init_mat(neural_network->layers[i], io_number_of_vectors);
			y_intermediate_outputs[i] = init_mat(neural_network->layers[i], io_number_of_vectors);
		}

		// copy training_input into y_intermediate_outputs[0]
		// to make y_0 == x_1
		// copy_batch(y_intermediate_outputs[0], training_input);
		copy_matrix(y_intermediate_outputs[0], training_input->data);

		// forward propagation
		for (int i = 1; i < number_of_layers; i++) {
			// l_i = W*x_i where (x_i == y_{i - 1})
			// multiply_batch_by_matrix(linear_intermediate_outputs[i]->data, weights[i - 1], y_intermediate_outputs[i - 1]->data);
			matrix_mult(linear_intermediate_outputs[i], neural_network->weights[i - 1], y_intermediate_outputs[i - 1]);

			// z_i = l_i + b_i
			// add_vector_to_batch(z_intermediate_outputs[i], linear_intermediate_outputs[i], biases[i - 1]);
			add_vector_to_matrix(z_intermediate_outputs[i], linear_intermediate_outputs[i], neural_network->biases[i - 1]);

			// y_i = f(z_i)
			// nonlinear_transform(y_intermediate_outputs[1], z_intermediate_outputs[1]);
			nonlinear_transform_mat(y_intermediate_outputs[i], z_intermediate_outputs[i]);
		}

		#ifdef ML_LIB_DEBUG_MODE
		// calculate error
		number sum = 0;
		for (int i = 0; i < training_output->data->number_of_rows; i++) {
			for (int j = 0; j < training_output->data->number_of_cols; j++) {
				sum += (VALUE_AT(training_output->data, i, j) - VALUE_AT(y_intermediate_outputs[number_of_layers - 1], i, j)) * (VALUE_AT(training_output->data, i, j) - VALUE_AT(y_intermediate_outputs[number_of_layers - 1], i, j));
			}
		}
		sum /= io_number_of_vectors;
		fprintf(stdout, "Error so far: %f\n", sum);
		#endif

		// backward propagation
		// batch* layer_output = training_output;
		matrix* layer_output = training_output->data;
		for (int j = number_of_layers - 1; j > 0; j--) {
			/*
			batch* dE_dy = create_empty_batch(layer_output->number_of_vectors, layer_output->vector_size);
			batch* dy_dz = create_empty_batch(layer_output->number_of_vectors, layer_output->vector_size);
			batch* dE_dz = create_empty_batch(layer_output->number_of_vectors, layer_output->vector_size);
			*/
			matrix* dE_dy = init_mat(layer_output->number_of_rows,layer_output->number_of_cols);
			matrix* dy_dz = init_mat(layer_output->number_of_rows,layer_output->number_of_cols);
			matrix* dE_dz = init_mat(layer_output->number_of_rows,layer_output->number_of_cols);

			matrix* grad_w = init_mat(neural_network->weights[j - 1]->number_of_rows, neural_network->weights[j - 1]->number_of_cols);
			vector* grad_b = init_vec(neural_network->biases[j - 1]->size);

			// dE/dy = y_intermediate_outputs[j] - y_theoretical_outputs[j]
			// auxillary_function_five(dE_dy, y_intermediate_outputs[j], layer_output, 1);
			matrix_sub(dE_dy, y_intermediate_outputs[j], layer_output);

			// dy/dz = f'(y_intermediates_outputs[j - 1]) or f'(x)
			// nonlinear_transform_derivative(dy_dz, y_intermediate_outputs[j - 1]);
			nonlinear_transform_derivative_mat(dy_dz, z_intermediate_outputs[j - 1]);

			// dE/dz = dE/dy . dy/dz
			// batch_hadamard_product(dE_dz, dE_dy, dy_dz);
			matrix_entrywise_product(dE_dz, dE_dy, dy_dz);

			// grad_w = dE_dz * transpose(y_intermediate_outputs[j - 1])
			// auxillary_function_one(grad_w, dE_dz, y_intermediate_outputs[j - 1], neural_network->gamma);
			// auxillary_function_two(grad_b, dE_dz, neural_network->gamma);
			matrix* x_intermediate_transpose = init_mat(y_intermediate_outputs[j - 1]->number_of_cols, y_intermediate_outputs[j - 1]->number_of_rows);
			matrix_transpose(x_intermediate_transpose, y_intermediate_outputs[j - 1]);
			matrix_scale(x_intermediate_transpose, x_intermediate_transpose, neural_network->gamma);
			matrix_mult(grad_w, dE_dz, x_intermediate_transpose);
			del_mat(x_intermediate_transpose);

			matrix_col_sum(grad_b, dE_dz);
			vector_scale(grad_b, grad_b, neural_network->gamma / io_number_of_vectors);
			fprintf(stdout, "----------\n");
			for (int x = 0; x < grad_w->number_of_rows; x++) {
				for (int y = 0; y < grad_w->number_of_cols; y++) {
					fprintf(stdout, "%f ", VALUE_AT(grad_w, x, y));
				}
				fprintf(stdout, "\n");
			}
			fprintf(stdout, "----------\n");
			for (int x = 0; x < grad_b->size; x++) {
				fprintf(stdout, "%f ", grad_b->v[x]);
			}
			fprintf(stdout, "\n----------\n");

			
			// update weights and biases
			// auxillary_function_three(neural_network->weights[j], grad_w);
			// auxillary_function_four(neural_network->biases[j], grad_b);

			matrix_sub(neural_network->weights[j - 1], neural_network->weights[j - 1], grad_w);
			vector_sub(neural_network->biases[j - 1], neural_network->biases[j - 1], grad_b);

			if (j != 1) {
				// batch* dE_dx = create_empty_batch(layer_output->number_of_vectors, y_intermediate_outputs[j - 1]->vector_size);
				matrix* dE_dx = init_mat(y_intermediate_outputs[j - 1]->number_of_rows, layer_output->number_of_cols);

				// multiply_batch_by_matrix(dE_dx, neural_network->weights[j], dE_dz);
				// auxillary_function_five(y_intermediate_outputs[j - 1], y_intermediate_outputs[j - 1], dE_dx, neural_network->gamma);
				matrix_mult(dE_dx, neural_network->weights[j - 1], dE_dz);
				matrix_scale(dE_dx, dE_dx, neural_network->gamma / io_number_of_vectors);
				matrix_sub(y_intermediate_outputs[j - 1], y_intermediate_outputs[j - 1], dE_dx);

				layer_output = y_intermediate_outputs[j - 1];

				// delete_batch(dE_dx);
				del_mat(dE_dx);
			}

			/*
			delete_batch(dE_dy);
			delete_batch(dy_dz);
			delete_batch(dE_dz);
			*/
			del_mat(dE_dy);
			del_mat(dy_dz);
			del_mat(dE_dz);
			
			del_mat(grad_w);
			del_vec(grad_b);
		}
		

		// delete the intermediate batches
		for (int i = 0; i < neural_network->number_of_layers; i++) {
			/*
			delete_batch(linear_intermediate_outputs[i]);
			delete_batch(z_intermediate_outputs[i]);
			delete_batch(y_intermediate_outputs[i]);
			*/
			del_mat(linear_intermediate_outputs[i]);
			del_mat(z_intermediate_outputs[i]);
			del_mat(y_intermediate_outputs[i]);
		}
		free(linear_intermediate_outputs);
		free(z_intermediate_outputs);
		free(y_intermediate_outputs);
		

		curr_nloops++;
	}


}


























/* 	AUXILLARY FUNCTIONS FOR THE TRAINING AND TESTING OF ARTIFICIAL NEURAL NETWORKS.  */





























/**
 * This function returns the matrix multiplication of dE/dz with the transpose of y, divided by the number of vectors in dE/dz (= # in y)
 */
void auxillary_function_one(matrix* grad_w, batch* dE_dz, batch* y, number gamma) {
	#ifdef ML_LIB_DEBUG_MODE
	if (dE_dz->number_of_vectors != y->number_of_vectors) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION ONE: The number of vectors do not match for some cursed reason.\n");
		exit(EXIT_FAILURE);
	}

	if (dE_dz->vector_size != grad_w->number_of_rows) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION ONE: The vector size of dE/dz does not match the number of rows in the gradient of the weight.\n");
		exit(EXIT_FAILURE);
	}

	if (y->vector_size != grad_w->number_of_cols) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION ONE: The vector size of y does not match the number of columns in the gradient of weight.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	size_t number_of_vectors = y->number_of_vectors;
	// matrix* y_transpose = tranpose(y);

	for (int i = 0; i < grad_w->number_of_rows; i++) {
		for (int j = 0; j < grad_w->number_of_cols; j++) {
			grad_w->m[i * grad_w->number_of_cols + j] = 0;
			for (int k = 0; k < number_of_vectors; k++) {
				grad_w->m[i * grad_w->number_of_cols + j] += dE_dz->data->m[i *dE_dz->data->number_of_cols + k] * y->data->m[k * y->data->number_of_cols + j];
			}
			grad_w->m[i * grad_w->number_of_cols + j] /= number_of_vectors;
			grad_w->m[i * grad_w->number_of_cols + j] *= -gamma;
		}
	}
}

/**
 * This function averages out all the columns in dE/dz and stores it in grad_b
 */
void auxillary_function_two(vector* grad_b, batch* dE_dz, number gamma) {
	#ifdef ML_LIB_DEBUG_MODE
	if (dE_dz->vector_size != grad_b->size) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION TWO: The sizes do not match for some cursed reason.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < dE_dz->vector_size; i++) {
		grad_b->v[i] = 0;
		for (int j = 0; j < dE_dz->number_of_vectors; j++) {
			grad_b->v[i] += dE_dz->data->m[i * dE_dz->number_of_vectors + j];
		}
		grad_b->v[i] /= dE_dz->number_of_vectors;
		grad_b->v[i] *= -gamma;
	}
}


void auxillary_function_three(matrix* weights, matrix* grad_w) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((weights->number_of_cols != grad_w->number_of_cols) || 
		(weights->number_of_rows != grad_w->number_of_rows)) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION THREE: The dimensions of the matrices do not match for some cursed reason.\n");
		exit(EXIT_FAILURE);
	}
	#endif
	
	for (int i = 0; i < weights->number_of_rows; i++) {
		for (int j = 0; j < weights->number_of_cols; j++) {
			weights->m[i * weights->number_of_cols + j] += grad_w->m[i * weights->number_of_cols + j];
		}
	}
}

void auxillary_function_four(vector* biases, vector* grad_b) {
	#ifdef ML_LIB_DEBUG_MODE
	if (biases->size != grad_b->size) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION FOUR: The vector sizes do not match for some cursed reason.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < biases->size; i++) {
		biases->v[i] += grad_b->v[i];
	}
}

/**
 * Subtraction of batches: out = first - gamma * second
 */
void auxillary_function_five(batch* out, batch* first, batch* second, number gamma) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((first->number_of_vectors != second->number_of_vectors) || (first->vector_size != second->vector_size)) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION FIVE: The dimensions of the input batches do not match for some cursed reason.\n");
		exit(EXIT_FAILURE);
	}

	if ((first->number_of_vectors != out->number_of_vectors) || (first->vector_size != out->vector_size)) {
		fprintf(stderr, "ERROR IN AUXILLARY FUNCTION FIVE: The dimensions of the output batch does not match the input batch.\n");
		exit(EXIT_FAILURE);
	}
	#endif

	for (int i = 0; i < first->vector_size; i++) {
		for (int j = 0; j < first->number_of_vectors; j++) {
			out->data->m[i * out->number_of_vectors + j]  = first->data->m[i * first->number_of_vectors + j];
			out->data->m[i * out->number_of_vectors + j] -= gamma * second->data->m[i * second->number_of_vectors + j];
		}
	}
}


/**
 * Copy the contents of the input batch into the output batch
 */
void copy_batch(batch* out, batch* in) {
	#ifdef ML_LIB_DEBUG_MODE
	if ((out->vector_size != in->vector_size) || (out->number_of_vectors != in->number_of_vectors)) {
		fprintf(stderr, "ERROR IN COPY BATCH: The dimensions of the batches do not match for some cursed reason.\n");
		exit(EXIT_FAILURE);
	}
	#endif
	
	for (int i = 0; i < out->vector_size; i++) {
		for (int j = 0; j < out->number_of_vectors; j++) {
			out->data->m[i * out->number_of_vectors + j] = in->data->m[i * out->number_of_vectors + j];
		}
	}
}