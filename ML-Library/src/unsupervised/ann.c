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
	neural_network->gamma = 0.4;

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
		exit(EXIT_FAILURE);
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
		exit(EXIT_FAILURE);
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
		fprintf(stderr, "ERROR IN NONLINEAR TRANSFORM DERIVATIVE DERIVATIVE: output batch does not match input batch");
		exit(EXIT_FAILURE);
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
		fprintf(stderr, "ERROR IN NONLINEAR TRANSFORM DERIVATIVE: output batch does not match input batch.\n");
		fprintf(stderr, "output: (%lu x %lu)\n", output->number_of_rows, output->number_of_cols);
		fprintf(stderr, "input: (%lu x %lu)\n", input->number_of_rows, input->number_of_cols);
		exit(EXIT_FAILURE);
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


	int nloops = 1;
	int idx = 0;
	int curr_nloops = 0;
	while (curr_nloops < nloops * 2) { //many_batches_training_input->number_of_batches
		batch* training_input = many_batches_training_input->ray_of_batches[idx % many_batches_training_input->number_of_batches];
		batch* training_output = many_batches_training_output->ray_of_batches[idx % many_batches_training_output->number_of_batches];
		idx = idx + 1;

		matrix** linear_intermediate_outputs;
		matrix** z_intermediate_outputs;
		matrix** y_intermediate_outputs;

		size_t io_number_of_vectors = training_input->number_of_vectors;

		#ifdef ML_LIB_DEBUG_MODE
		linear_intermediate_outputs = (matrix **)calloc(neural_network->number_of_layers, sizeof(matrix *));
		y_intermediate_outputs = (matrix **)calloc(neural_network->number_of_layers, sizeof(matrix *));
		z_intermediate_outputs = (matrix **)calloc(neural_network->number_of_layers, sizeof(matrix *));
		#else
		linear_intermediate_outputs = (matrix **)malloc(neural_network->number_of_layers * sizeof(matrix *));
		y_intermediate_outputs = (matrix **)malloc(neural_network->number_of_layers * sizeof(matrix *));
		z_intermediate_outputs = (matrix **)malloc(neural_network->number_of_layers * sizeof(matrix *));
		#endif


		for (int i = 0; i < number_of_layers; i++) {
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

		/*
		for (int idx = 0; idx < number_of_layers - 1; idx++) {
			fprintf(stdout, "----------\n");
			for (int x = 0; x < neural_network->weights[idx]->number_of_rows; x++) {
				for (int y = 0; y < neural_network->weights[idx]->number_of_cols; y++) {
					fprintf(stdout, "%f ", VALUE_AT(neural_network->weights[idx], x, y));
				}
				fprintf(stdout, "\n");
			}
			fprintf(stdout, "----------\n");
		}
		*/
		

		#ifdef ML_LIB_DEBUG_MODE
		// calculate error
		number sum = 0;
		for (int x = 0; x < training_output->data->number_of_rows; x++) {
			for (int y = 0; y < training_output->data->number_of_cols; y++) {
				sum += (VALUE_AT(training_output->data, x, y) - VALUE_AT(y_intermediate_outputs[number_of_layers - 1], x, y)) * (VALUE_AT(training_output->data, x, y) - VALUE_AT(y_intermediate_outputs[number_of_layers - 1], x, y));
			}
		}
		sum /= io_number_of_vectors;
		fprintf(stdout, "Error so far: %f\n", sum);
		#endif

		// backward propagation
		// batch* layer_output = training_output;
		matrix* layer_output = training_output->data;
		for (int j = number_of_layers - 1; j > 0; j--) {
			fprintf(stdout, "j: %d\n", j);

			matrix* dE_dy = init_mat(layer_output->number_of_rows,layer_output->number_of_cols);
			matrix* dy_dz = init_mat(layer_output->number_of_rows,layer_output->number_of_cols);
			matrix* dE_dz = init_mat(layer_output->number_of_rows,layer_output->number_of_cols);

			matrix* grad_w = init_mat(neural_network->weights[j - 1]->number_of_rows, neural_network->weights[j - 1]->number_of_cols);
			vector* grad_b = init_vec(neural_network->biases[j - 1]->size);

			// dE/dy = y_intermediate_outputs[j] - y_theoretical_outputs[j]
			if (j == number_of_layers - 1) {
				matrix_sub(dE_dy, y_intermediate_outputs[j], layer_output);
			} else {
				copy_matrix(dE_dy, layer_output);
			}
			// dy/dz = f'(y_intermediates_outputs[j - 1]) or f'(x)
			// nonlinear_transform_derivative(dy_dz, y_intermediate_outputs[j - 1]);
			nonlinear_transform_derivative_mat(dy_dz, z_intermediate_outputs[j]);

			// dE/dz = dE/dy . dy/dz
			// batch_hadamard_product(dE_dz, dE_dy, dy_dz);
			matrix_entrywise_product(dE_dz, dE_dy, dy_dz);
			
			// grad_w = dE_dz * transpose(y_intermediate_outputs[j - 1])
			// auxillary_function_one(grad_w, dE_dz, y_intermediate_outputs[j - 1], neural_network->gamma);
			// auxillary_function_two(grad_b, dE_dz, neural_network->gamma);
			matrix* x_intermediate_transpose = init_mat(y_intermediate_outputs[j - 1]->number_of_cols, y_intermediate_outputs[j - 1]->number_of_rows);
			matrix_transpose(x_intermediate_transpose, y_intermediate_outputs[j - 1]);
			matrix_mult(grad_w, dE_dz, x_intermediate_transpose);
			del_mat(x_intermediate_transpose);

			matrix_col_sum(grad_b, dE_dz);
			
			/*
			fprintf(stdout, "----------\ngrad_w\n");
			for (int x = 0; x < grad_w->number_of_rows; x++) {
				for (int y = 0; y < grad_w->number_of_cols; y++) {
					fprintf(stdout, "%f ", VALUE_AT(grad_w, x, y));
				}
				fprintf(stdout, "\n");
			}
			fprintf(stdout, "----------\ngrad_b\n");
			for (int x = 0; x < grad_b->size; x++) {
				fprintf(stdout, "%f ", grad_b->v[x]);
			}
			fprintf(stdout, "\n----------\n");
			*/
			

			
			// update weights and biases
			// auxillary_function_three(neural_network->weights[j], grad_w);
			// auxillary_function_four(neural_network->biases[j], grad_b);

			matrix_scale(grad_w, grad_w, neural_network->gamma / io_number_of_vectors);
			vector_scale(grad_b, grad_b, neural_network->gamma / io_number_of_vectors);

			matrix_sub(neural_network->weights[j - 1], neural_network->weights[j - 1], grad_w);
			vector_sub(neural_network->biases[j - 1], neural_network->biases[j - 1], grad_b);

			if (j != 1) {
				// batch* dE_dx = create_empty_batch(layer_output->number_of_vectors, y_intermediate_outputs[j - 1]->vector_size);
				matrix* dE_dx = init_mat(y_intermediate_outputs[j - 1]->number_of_rows, layer_output->number_of_cols);
				matrix* weight_transpose = init_mat(neural_network->weights[j - 1]->number_of_cols, neural_network->weights[j - 1]->number_of_rows);
				matrix_transpose(weight_transpose, neural_network->weights[j - 1]);

				// multiply_batch_by_matrix(dE_dx, neural_network->weights[j], dE_dz);
				// auxillary_function_five(y_intermediate_outputs[j - 1], y_intermediate_outputs[j - 1], dE_dx, neural_network->gamma);
				
				matrix_mult(dE_dx, weight_transpose, dE_dz);
				matrix_scale(dE_dx, dE_dx, neural_network->gamma / io_number_of_vectors);

				// set layer_output to be dE_dx
				if (j != number_of_layers - 1) {
					del_mat(layer_output);
				}
				layer_output = init_mat(dE_dx->number_of_rows, dE_dx->number_of_cols);
				copy_matrix(layer_output, dE_dx);

				fprintf(stdout, "----------\ndE/dx\n");
				for (int x = 0; x < dE_dx->number_of_rows; x++) {
					for (int y = 0; y < dE_dx->number_of_cols; y++) {
						fprintf(stdout, "%f ", VALUE_AT(dE_dx, x, y));
					}
					fprintf(stdout, "\n");
				}
				fprintf(stdout, "----------\n");
				
				// ;ADFJSLKFDSAIOFJPASDJFLKSAD;NJFAPSLDI
				// THIS WAS THE PROBLEM!!!!!
				// layer_output = y_intermediate_outputs[j - 1];

				// delete_batch(dE_dx);
				del_mat(dE_dx);
			} else {
				// delete final layer matrix
				del_mat(layer_output);
			}


			del_mat(dE_dy);
			del_mat(dy_dz);
			del_mat(dE_dz);
			
			del_mat(grad_w);
			del_vec(grad_b);
		}

		// delete the intermediate batches
		for (int i = 0; i < neural_network->number_of_layers; i++) {
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



















