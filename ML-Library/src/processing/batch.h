#include "../mllib.h"
#include "../math/matrix.h"

#ifndef MLLIB_BATCH_H
#define MLLIB_BATCH_H

struct batch_ {
	matrix* data;
	size_t number_of_vectors;
	size_t vector_size; // vector_size == data[i]->size for all i
};
typedef struct batch_ batch;

struct m_batch_ {
	batch** ray_of_batches;
	size_t number_of_batches;
	size_t total_number_of_vectors;
	size_t vector_size;
};
typedef struct m_batch_ m_batch;

/**
 * Batch initialization, deletion, and loading
 */
batch* create_empty_batch(size_t number_of_vectors, size_t vec_size);
void delete_batch(batch* batch_to_delete);
void load_data_into_batch(batch* empty_batch, vector** huge_number_of_data, size_t number_of_data);


/**
 * Batch operations
 */
void multiply_batch_by_matrix(batch *output_batch_vectors, matrix *mat, batch *input_batch_vectors);
void add_vector_to_batch(batch* output_batch, batch* input_batch, vector* vec);
void batch_hadamard_product(batch* output_batch, batch* product_one, batch* product_two);

#endif