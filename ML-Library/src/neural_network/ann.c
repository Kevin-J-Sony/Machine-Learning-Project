#include "ann.h"

/*
ann* initialize_ann(size_t* sizes, size_t number_of_layers, boolean classification_task) {
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
				VALUE_AT(neural_network->weights[i], j, k) = ((float)rand())/(RAND_MAX) - 0.5; //1;
			}
			neural_network->biases[i]->v[j] = 0;
		}

		neural_network->layers[i] = sizes[i];
	}
	neural_network->layers[number_of_layers - 1] = sizes[number_of_layers - 1];
	neural_network->number_of_layers = number_of_layers;
	neural_network->gamma = 0.005;
	neural_network->is_classifier = classification_task;

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
*/

NeuralNet* initialize_nn(const size_t* dims, size_t dims_ray_size, number gamma) {
	NeuralNet* nn = malloc(sizeof(*nn));
	
	nn->number_of_layers = dims_ray_size;
	nn->gamma = gamma;
	#ifdef ML_LIB_DEBUG_MODE
	nn->layers = calloc(nn->number_of_layers, sizeof(ann_layer));
	#else
	nn->layers = malloc(nn->number_of_layers * sizeof(ann_layer));
	#endif

	for (int i = 0; i < nn->number_of_layers; i++) {
		size_t in = dims[i];
		size_t out = dims[i + 1];
		nn->layers[i].dim_input = in;
		nn->layers[i].dim_output = out;

		nn->layers[i].W = init_rand_mat(out, in);
		nn->layers[i].b = init_vec(out);

		// since for now we're doing classification task
		if (i == nn->number_of_layers - 1) {
			nn->layers[i].a = sigmoid;
			nn->layers[i].a_prime = sigmoid_derivative;
		} else {
			nn->layers[i].a = leaky_relu;
			nn->layers[i].a_prime = leaky_relu_derivative;
		}
	}
	
	return nn;
}

void free_nn(NeuralNet* nn);



/**
 * Training function for the neural network. Accepts a batch of inputs and a batch of outputs.
 */
void train(NeuralNet* nn, m_batch* many_batches_training_input, m_batch* many_batches_training_output, size_t number_of_loops) {
	#ifdef ML_LIB_DEBUG_MODE
	if (many_batches_training_input->total_number_of_vectors != many_batches_training_output->total_number_of_vectors) {
		fprintf(stderr, "ANN TRAINING ERROR: Number of inputs does not match number of outputs\n");
		exit(EXIT_FAILURE);
	}
	if (many_batches_training_input->vector_size != nn->layers[0].dim_input) {
		fprintf(stderr, "ANN TRAINING ERROR: Size of inputs do not match input layer of neural network\n");
		exit(EXIT_FAILURE);
	}
	if (many_batches_training_output->vector_size != nn->layers[nn->number_of_layers - 1].dim_output) {
		fprintf(stderr, "ANN TRAINING ERROR: Size of outputs does not match output layer of neural network\n");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < many_batches_training_input->number_of_batches; i++) {
		if ((many_batches_training_input->ray_of_batches[0]->number_of_vectors != many_batches_training_input->ray_of_batches[i]->number_of_vectors) ||
			(many_batches_training_output->ray_of_batches[0]->number_of_vectors != many_batches_training_output->ray_of_batches[i]->number_of_vectors)) {
			fprintf(stderr, "ANN TRAINING ERROR: Inconsistent batch sizes.\n");
			exit(EXIT_FAILURE);
		}
	}
	#endif

	size_t number_of_layers = nn->number_of_layers;

	matrix** linear_intermediate_outputs;
	matrix** z_intermediate_outputs;
	matrix** y_intermediate_outputs;

	#ifdef ML_LIB_DEBUG_MODE
	linear_intermediate_outputs = (matrix **)calloc(nn->number_of_layers + 1, sizeof(matrix *));
	y_intermediate_outputs = (matrix **)calloc(nn->number_of_layers + 1, sizeof(matrix *));
	z_intermediate_outputs = (matrix **)calloc(nn->number_of_layers + 1, sizeof(matrix *));
	#else
	linear_intermediate_outputs = (matrix **)malloc((nn->number_of_layers + 1) * sizeof(matrix *));
	y_intermediate_outputs = (matrix **)malloc((nn->number_of_layers + 1) * sizeof(matrix *));
	z_intermediate_outputs = (matrix **)malloc((nn->number_of_layers + 1) * sizeof(matrix *));
	#endif

	size_t nloops = number_of_loops;
	int idx = 0;
	int curr_nloops = 0;
	number total_loss = 0;
	while (curr_nloops < nloops * many_batches_training_input->number_of_batches) { 
		batch* training_input = many_batches_training_input->ray_of_batches[idx % many_batches_training_input->number_of_batches];
		batch* training_output = many_batches_training_output->ray_of_batches[idx % many_batches_training_output->number_of_batches];

		size_t io_number_of_vectors = training_input->number_of_vectors;

		linear_intermediate_outputs[i] = init_mat(nn->layers[i - 1].dim_output, io_number_of_vectors);
		z_intermediate_outputs[i] = init_mat(nn->layers[i - 1].dim_output, io_number_of_vectors);
		y_intermediate_outputs[i] = init_mat(nn->layers[i - 1].dim_output, io_number_of_vectors);
	
		for (int i = 1; i < number_of_layers + 1; i++) {
			linear_intermediate_outputs[i] = init_mat(nn->layers[i - 1].dim_output, io_number_of_vectors);
			z_intermediate_outputs[i] = init_mat(nn->layers[i - 1].dim_output, io_number_of_vectors);
			y_intermediate_outputs[i] = init_mat(nn->layers[i - 1].dim_output, io_number_of_vectors);
		}
		
		idx = idx + 1;

		// copy training_input into y_intermediate_outputs[0]
		// to make x_0 = y_0
		copy_matrix(y_intermediate_outputs[0], training_input->data);

		// forward propagation
		for (int i = 0; i < number_of_layers; i++) {
			// l_i = W*x_i where (x_i == y_{i - 1})
			matrix_mult(linear_intermediate_outputs[i], neural_network->neura, y_intermediate_outputs[i - 1]);

			// z_i = l_i + b_i
			add_vector_to_matrix(z_intermediate_outputs[i], linear_intermediate_outputs[i], neural_network->biases[i - 1]);

			// y_i = f(z_i)
			leaky_relu(y_intermediate_outputs[i], z_intermediate_outputs[i]);
		}

		number error_rate = io_number_of_vectors;
		// in the case the neural network is a classifier, we want the entries to be either a 0 or 1
		if (neural_network->is_classifier) {
			matrix* pred_output = y_intermediate_outputs[number_of_layers - 1];

			for (int curr_input_idx = 0; curr_input_idx < pred_output->number_of_cols; curr_input_idx++) {
				int largest_y_idx = 0;
				number t1, t2;
				for (int y = 1; y < pred_output->number_of_rows; y++) {
					t1 = VALUE_AT(pred_output, y, curr_input_idx);
					t1 = (t1 > 0) ? t1 : -t1;

					t2 = VALUE_AT(pred_output, largest_y_idx, curr_input_idx);
					t2 = (t2 > 0) ? t2 : -t1;
					if (t1 > t2) {
						largest_y_idx = y;
					}
				}

				for (int y = 0; y < pred_output->number_of_rows; y++) {
					VALUE_AT(pred_output, y, curr_input_idx) = (y == largest_y_idx) ? 1 : 0;
				}
				
				if (VALUE_AT(training_output->data, largest_y_idx, curr_input_idx) != VALUE_AT(pred_output, largest_y_idx, curr_input_idx)) {
					error_rate -= 1;
				}
			}
		}
		error_rate /= io_number_of_vectors;

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
		number loss = 0;
		for (int x = 0; x < training_output->data->number_of_rows; x++) {
			for (int y = 0; y < training_output->data->number_of_cols; y++) {
				loss += (VALUE_AT(training_output->data, x, y) - VALUE_AT(y_intermediate_outputs[number_of_layers - 1], x, y)) * (VALUE_AT(training_output->data, x, y) - VALUE_AT(y_intermediate_outputs[number_of_layers - 1], x, y));
			}
		}
		// losos differs from error, since loss measures how far the prediction is from the output
		loss /= io_number_of_vectors;
		total_loss += loss;
		if (idx % many_batches_training_input->number_of_batches == 0) {
			total_loss /= many_batches_training_input->number_of_batches;
			fprintf(stdout, "Loop %d. Loss so far: %f. Error so far: %f. Gamma so far: %f)\n", idx/many_batches_training_input->number_of_batches, total_loss, error_rate * 100, neural_network->gamma);
			total_loss = 0;
		}
		#endif

		// backward propagation
		// batch* layer_output = training_output;
		matrix* prev_layer_err = training_output->data;
		for (int j = number_of_layers - 1; j > 0; j--) {
			matrix* dL_dy = init_mat(prev_layer_err->number_of_rows,prev_layer_err->number_of_cols);
			matrix* dy_dz = init_mat(prev_layer_err->number_of_rows,prev_layer_err->number_of_cols);
			matrix* delta = init_mat(prev_layer_err->number_of_rows,prev_layer_err->number_of_cols);

			matrix* grad_w = init_mat(neural_network->weights[j - 1]->number_of_rows, neural_network->weights[j - 1]->number_of_cols);
			vector* grad_b = init_vec(neural_network->biases[j - 1]->size);

			// dE/dy = y_intermediate_outputs[j] - y_theoretical_outputs[j]
			if (j == number_of_layers - 1) {
				matrix_sub(dL_dy, y_intermediate_outputs[j], prev_layer_err);
			} else {
				copy_matrix(dL_dy, prev_layer_err);
			}
			/*
			fprintf(stdout, "----------\ndE_dy at layer %d\n", j + 1);
			for (int x = 0; x < dE_dy->number_of_rows; x++) {
				for (int y = 0; y < dE_dy->number_of_cols; y++) {
					fprintf(stdout, "%f ", VALUE_AT(dE_dy, x, y));
				}
				fprintf(stdout, "\n");
			}
			fprintf(stdout, "----------\n");
			*/
			

			// dy/dz = f'(z)
			leaky_relu_derivative(dy_dz, z_intermediate_outputs[j]);

			// delta = dL/dy . dy/dz
			matrix_entrywise_product(delta, dL_dy, dy_dz);
			
			// dL/dw = delta * (x^{(i-1)})^T
			matrix* x_intermediate_transpose = init_mat(y_intermediate_outputs[j - 1]->number_of_cols, y_intermediate_outputs[j - 1]->number_of_rows);
			matrix_transpose(x_intermediate_transpose, y_intermediate_outputs[j - 1]);
			matrix_mult(grad_w, delta, x_intermediate_transpose);
			del_mat(x_intermediate_transpose);

			matrix_col_sum(grad_b, delta);
			
			matrix_scale(grad_w, grad_w, neural_network->gamma / io_number_of_vectors);
			vector_scale(grad_b, grad_b, neural_network->gamma / io_number_of_vectors);

			if (j != 1) {
				matrix* dL_dx = init_mat(y_intermediate_outputs[j - 1]->number_of_rows, prev_layer_err->number_of_cols);
				matrix* weight_transpose = init_mat(neural_network->weights[j - 1]->number_of_cols, neural_network->weights[j - 1]->number_of_rows);
				matrix_transpose(weight_transpose, neural_network->weights[j - 1]);


				// dL/dx = (W^{(i)})^T * delta
				matrix_mult(dL_dx, weight_transpose, delta);

				if (j != number_of_layers - 1) {
					del_mat(prev_layer_err);
				}

				prev_layer_err = init_mat(dL_dx->number_of_rows, dL_dx->number_of_cols);
				matrix_entrywise_product(prev_layer_err, dL_dx, y_intermediate_outputs[j - 1]);

				/*
				fprintf(stdout, "----------\nW[%d].T\n", j - 1);
				for (int x = 0; x < weight_transpose->number_of_rows; x++) {
					for (int y = 0; y < weight_transpose->number_of_cols; y++) {
						fprintf(stdout, "%f ", VALUE_AT(weight_transpose, x, y));
					}
					fprintf(stdout, "\n");
				}

				fprintf(stdout, "----------\ndE/dz\n");
				for (int x = 0; x < dE_dz->number_of_rows; x++) {
					for (int y = 0; y < dE_dz->number_of_cols; y++) {
						fprintf(stdout, "%f ", VALUE_AT(dE_dz, x, y));
					}
					fprintf(stdout, "\n");
				}

				fprintf(stdout, "----------\ndE/dx\n");
				for (int x = 0; x < dE_dx->number_of_rows; x++) {
					for (int y = 0; y < dE_dx->number_of_cols; y++) {
						fprintf(stdout, "%f ", VALUE_AT(dE_dx, x, y));
					}
					fprintf(stdout, "\n");
				}
				fprintf(stdout, "----------\n");
				*/
				
				// delete_batch(dE_dx);
				del_mat(weight_transpose);
				del_mat(dL_dx);
			} else {
				// delete final layer matrix
				del_mat(prev_layer_err);
			}

			matrix_sub(neural_network->weights[j - 1], neural_network->weights[j - 1], grad_w);
			vector_sub(neural_network->biases[j - 1], neural_network->biases[j - 1], grad_b);

			del_mat(dL_dy);
			del_mat(dy_dz);
			del_mat(delta);
			
			del_mat(grad_w);
			del_vec(grad_b);
		}		

		curr_nloops++;

		// delete the intermediate batches
		for (int i = 0; i < neural_network->number_of_layers; i++) {
			del_mat(linear_intermediate_outputs[i]);
			del_mat(z_intermediate_outputs[i]);
			del_mat(y_intermediate_outputs[i]);
		}

	}

	free(linear_intermediate_outputs);
	free(z_intermediate_outputs);
	free(y_intermediate_outputs);

}









batch* pass_forward(ann* neural_network, batch* inputs) {
	#ifdef ML_LIB_DEBUG_MODE
	if (inputs->vector_size != neural_network->layers[0]) {
		fprintf(stderr, "ANN PASS FORWARD: Size of inputs do not match input layer of neural network\n");
		exit(EXIT_FAILURE);
	}
	#endif

	size_t io_number_of_vectors = inputs->number_of_vectors;

	matrix** linear_intermediate_outputs;
	matrix** z_intermediate_outputs;
	matrix** y_intermediate_outputs;

	size_t number_of_layers = neural_network->number_of_layers;

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

	copy_matrix(y_intermediate_outputs[0], inputs->data);

	// forward propagation
	for (int i = 1; i < number_of_layers; i++) {
		// l_i = W*x_i where (x_i == y_{i - 1})
		matrix_mult(linear_intermediate_outputs[i], neural_network->weights[i - 1], y_intermediate_outputs[i - 1]);

		// z_i = l_i + b_i
		add_vector_to_matrix(z_intermediate_outputs[i], linear_intermediate_outputs[i], neural_network->biases[i - 1]);

		// y_i = f(z_i)
		leaky_relu(y_intermediate_outputs[i], z_intermediate_outputs[i]);
	}

	batch* predictions = create_empty_batch(y_intermediate_outputs[number_of_layers - 1]->number_of_cols, y_intermediate_outputs[number_of_layers - 1]->number_of_rows);

	copy_matrix(predictions->data, y_intermediate_outputs[number_of_layers - 1]);

	// delete the intermediate batches
	// this only works if the batch_size for all batches are the same
	for (int i = 0; i < neural_network->number_of_layers; i++) {
		del_mat(linear_intermediate_outputs[i]);
		del_mat(z_intermediate_outputs[i]);
		del_mat(y_intermediate_outputs[i]);
	}
	free(linear_intermediate_outputs);
	free(z_intermediate_outputs);
	free(y_intermediate_outputs);

	return predictions;
}

#ifdef ML_LIB_DEBUG_MODE
void print_network(ann* neural_network) {
	fprintf(stdout, "----------\nNeural network info\n");

	size_t number_of_layers = neural_network->number_of_layers;
	for (int i = 0; i < number_of_layers - 1; i++) {
		print_mat(neural_network->weights[i]);
		print_vec(neural_network->biases[i]);
	}
}
#endif