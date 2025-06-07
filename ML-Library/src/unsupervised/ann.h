#include "mllib.h"

#ifndef MLLIB_ANN_H
#define MLLIB_ANN_H

struct ann_bias_ {
    vector* b;
}
typedef struct ann_bias_ ann_bias;


struct ann_weight_ {
    matrix* w;
}
typedef struct ann_weight_ ann_weight;


struct ann_ {
    // get a list of biases (a list of vectors)
    // get a list of weights
    ann_bias* biases; // (vector** biases)
    ann_weight* weights; // (matrix** weights)

    int number_of_biases;
    int number_of_weights;

    int size_of_input;
    int size_of_output;
};
typedef struct ann_ ann;

#endif