import ctypes
from read_numbers import *

# Define the vector structure in ctypes
class Vector(ctypes.Structure):
    _fields_ = [
        ("v", ctypes.POINTER(ctypes.c_float)),
        ("size", ctypes.c_size_t)
    ]

class Matrix(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.POINTER(ctypes.c_float)),
        ("number_of_rows", ctypes.c_size_t),
        ("number_of_cols", ctypes.c_size_t)
    ]

class Batch(ctypes.Structure):
    __fields__ = [
        ("data", ctypes.POINTER(Matrix)),
        ("number_of_vectors", ctypes.c_size_t),
        ("vector_size", ctypes.c_size_t)
    ]
    ...

class ManyBatches(ctypes.Structure):
    __fields__ = [
        ("ray_of_batches", ctypes.POINTER(ctypes.POINTER(Batch))),
        ("number_of_batches", ctypes.c_size_t),
        ("total_number_of_vectors", ctypes.c_size_t),
        ("vector_size", ctypes.c_size_t)
    ]

class ArtificialNeuralNetwork(ctypes.Structure):
    __fields__ = [
        ("weights", ctypes.POINTER(ctypes.POINTER(Matrix))),
        ("biases", ctypes.POINTER(ctypes.POINTER(Vector))),
        ("layers", ctypes.POINTER(ctypes.c_size_t)),
        ("number_of_layers", ctypes.c_size_t),
        ("gamma", ctypes.c_float)
    ]


# Get the shared library
lib = ctypes.CDLL("../lib/libmymllib.so")

# List out all the methods in the shared library, ones that we will be explicitly using anyways
def initialize_ann(layers):
    lib.initialize_ann
    ...

def load_data_into_batches(data: list[list[float]], number_of_data_points: int, number_of_batches: int):
    # convert data from python equivalent to c
    x = (ctypes.POINTER(Vector) * number_of_data_points)()
    arg1 = ctypes.cast(x, ctypes.POINTER(ctypes.POINTER(Vector)))
    print(arg1)
    for i in range(number_of_data_points):
        for j in range(len(data[i])):
            arg1[i][j].value = data[i][j]
    
    arg2 = ctypes.c_size_t(number_of_data_points)
    arg3 = ctypes.c_size_t(number_of_batches)
    
    lib.load_data_into_batches.argtypes = [ctypes.POINTER(ctypes.POINTER(Vector)), ctypes.c_size_t, ctypes.c_size_t]
    lib.load_data_into_batches.restype = ctypes.POINTER(ManyBatches)
    
    mb = lib.load_data_into_batches(arg1, arg2, arg3)
    print(mb)
    return mb
    ...

def train(ann, training_inputs, training_outputs):
    ...
    
def pass_forward(ann, inputs):
    ...

def delete_batches(m_batches_to_delete):
    ...

def deallocate_ann(ann):
    ...


if __name__ == "__main__":
    train_input, train_label = read_train_data(1)
    load_data_into_batches(train_input, 1, 1)