#include "../mllib.h"


#ifdef ML_LIB_DEBUG_MODE
#include <stdio.h>
#include <stdlib.h>
#endif


#ifndef MLLIB_MATRIX_H
#define MLLIB_MATRIX_H

// define a data type 'number' to take on float, double, or long double
typedef double number;

// define a basic vector data structure
struct vector_ {
    int size;
    number* vector;
};
typedef struct vector_ vector;


struct matrix_ {
    int row;
    int col;
    number** matrix;
};
typedef struct matrix_ matrix;


// Functions to initialize vectors and matrices
vector* init_vec(int length);
matrix* init_mat(int row, int col);

// Functions to deallocate vectors and matrices
void free_vec(vector* v);
void free_mat(matrix* m);

// Get functions
number get_vec(vector* v, int index);
number get_mat(matrix* m, int row_index, int col_index);

// Set functions
void set_vec(vector* v, int index, int value);
void set_mat(matrix* m, int row_index, int col_index, int value);

// Functions for basic operation
vector* add(vector* a, vector* b);
matrix* mult(matrix* a, matrix* b);
vector* mult(matrix* a, vector* b);


#endif
