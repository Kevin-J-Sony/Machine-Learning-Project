#include "../mllib.h"
#include "../math/matrix.h"

#ifndef MLLIB_BATCH_H
#define MLLIB_BATCH_H

/**
 * Create a structure over data (represented as vectors). This is done to improve computing speeds.
 * More concretely, we could feed the neural network one input at a time to train it, but such a method is highly
 * inoptimal. If we have 1000 inputs for example, training one input at a time would mean the neural network would have to
 * iterate 1000 times. If instead, the neural network can accept 20 inputs at a time, it would only have to iterate
 * 50 times. It's a massive reduction, but requires additional scaffolding.
 * 
 * Of course, the benefits are massively reaped only when CUDA is employed.
 */
struct batch_ {
    vector** data; // matrix* data;
    size_t number_of_vectors;
    size_t batch_size;
    size_t vector_size; // vector_size == data[i]->size for all i
};
typedef struct batch_ batch;

/**
 * Batch initialization, deletion, and loading
 */
batch* create_empty_batch(size_t number_of_vectors, size_t b_size, size_t vec_size);
void delete_batch(batch* batch_to_delete);
void load_data_into_batch(batch* empty_batch, vector** huge_number_of_data, size_t number_of_data, size_t batch_size);


/**
 * Batch operations
 */
void multiply_single_batch(batch *output_sbatch_of_vectors, matrix *mat, batch *sbatch_of_vectors);

#endif