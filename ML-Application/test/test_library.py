import ctypes
import numpy as np
from read_numbers import read_train_data, read_test_data

# --- C struct definitions ---
class Vector(ctypes.Structure):
    _fields_ = [
        ('v', ctypes.POINTER(ctypes.c_float)),
        ('size', ctypes.c_size_t),
    ]

class Matrix(ctypes.Structure):
    _fields_ = [
        ('m', ctypes.POINTER(ctypes.c_float)),
        ('number_of_rows', ctypes.c_size_t),
        ('number_of_cols', ctypes.c_size_t),
    ]

class Batch(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(Matrix)),
        ('number_of_vectors', ctypes.c_size_t),
        ('vector_size', ctypes.c_size_t),
    ]

class ManyBatches(ctypes.Structure):
    _fields_ = [
        ('ray_of_batches', ctypes.POINTER(ctypes.POINTER(Batch))),
        ('number_of_batches', ctypes.c_size_t),
        ('total_number_of_vectors', ctypes.c_size_t),
        ('vector_size', ctypes.c_size_t),
    ]

class ArtificialNeuralNetwork(ctypes.Structure):
    _fields_ = [
        ('weights', ctypes.POINTER(ctypes.POINTER(Matrix))),
        ('biases', ctypes.POINTER(ctypes.POINTER(Vector))),
        ('layers', ctypes.POINTER(ctypes.c_size_t)),
        ('number_of_layers', ctypes.c_size_t),
        ('gamma', ctypes.c_float),
    ]

# Load the dynamic library
#lib = ctypes.CDLL(r"../lib/libmymllib.so")
lib = ctypes.CDLL(r"../lib/train.dll")

# --- Function prototypes ---
lib.initialize_ann.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
lib.initialize_ann.restype = ctypes.POINTER(ArtificialNeuralNetwork)

lib.deallocate_ann.argtypes = [ctypes.POINTER(ArtificialNeuralNetwork)]
lib.deallocate_ann.restype = None

lib.load_data_into_batches.argtypes = [ctypes.POINTER(ctypes.POINTER(Vector)), ctypes.c_size_t, ctypes.c_size_t]
lib.load_data_into_batches.restype = ctypes.POINTER(ManyBatches)

lib.delete_batches.argtypes = [ctypes.POINTER(ManyBatches)]
lib.delete_batches.restype = None

lib.train.argtypes = [
    ctypes.POINTER(ArtificialNeuralNetwork),
    ctypes.POINTER(ManyBatches),
    ctypes.POINTER(ManyBatches),
    ctypes.c_size_t
]
lib.train.restype = None

lib.pass_forward.argtypes = [ctypes.POINTER(ArtificialNeuralNetwork), ctypes.POINTER(Batch)]
lib.pass_forward.restype = ctypes.POINTER(Batch)

# --- Python wrappers ---
def initialize_ann(layer_sizes: list[int]) -> ctypes.POINTER(ArtificialNeuralNetwork):
    n = len(layer_sizes)
    arr = (ctypes.c_size_t * n)(*layer_sizes)
    return lib.initialize_ann(arr, ctypes.c_size_t(n))


def load_data_into_batches(data: list[list[float]], num_data: int, num_batches: int) -> ctypes.POINTER(ManyBatches):
    # Allocate array of Vector* pointers
    vec_array = (ctypes.POINTER(Vector) * num_data)()
    for i, vec in enumerate(data):
        v = Vector()
        arr = (ctypes.c_float * len(vec))(*vec)
        v.v = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))
        v.size = ctypes.c_size_t(len(vec))
        vec_array[i] = ctypes.pointer(v)
    return lib.load_data_into_batches(vec_array, ctypes.c_size_t(num_data), ctypes.c_size_t(num_batches))


def train(ann: ctypes.POINTER(ArtificialNeuralNetwork),
          inputs: ctypes.POINTER(ManyBatches),
          outputs: ctypes.POINTER(ManyBatches),
          loops: ctypes.c_size_t) -> None:
    lib.train(ann, inputs, outputs, loops)


def pass_forward(ann: ctypes.POINTER(ArtificialNeuralNetwork),
                 many_batches: ctypes.POINTER(ManyBatches)) -> np.ndarray:
    mb = many_batches.contents
    # Get first batch pointer
    first_batch_ptr = mb.ray_of_batches[0]
    out_batch_ptr = lib.pass_forward(ann, first_batch_ptr)
    out_batch = out_batch_ptr.contents
    mat = out_batch.data.contents
    # Convert to numpy array
    shape = (mat.number_of_rows, mat.number_of_cols)
    buf = ctypes.cast(mat.m, ctypes.POINTER(ctypes.c_float * (shape[0] * shape[1]))).contents
    arr = np.frombuffer(buf, dtype=np.float32).reshape(shape)
    return arr.T


def delete_batches(mb: ctypes.POINTER(ManyBatches)) -> None:
    lib.delete_batches(mb)


def deallocate_ann(ann: ctypes.POINTER(ArtificialNeuralNetwork)) -> None:
    lib.deallocate_ann(ann)


# --- Example usage ---
if __name__ == '__main__':
    #import tracemalloc
    #tracemalloc.start()
    import faulthandler
    faulthandler.enable()
    
    # Read and batch data
    train_inputs, train_labels = read_train_data(4000)
    
    mb_in = load_data_into_batches(train_inputs, 4000, 16)
    mb_out = load_data_into_batches(train_labels, 4000, 16)

    # Initialize network
    ann = initialize_ann([784, 128, 10])

    # Train
    train(ann, mb_in, mb_out, 50)

    # Read and batch test data
    test_inputs, test_labels = read_train_data(20)
    
    mb_test_in = load_data_into_batches(test_inputs, 20, 20)

    # Inference
    preds = pass_forward(ann, mb_test_in)
    
    print(type(preds))
    print('Predictions shape:', preds.shape)
    for i in range(len(preds)):
        print(preds[i])
        print(test_labels[i])
        print("\n\n")

    # Cleanup
    delete_batches(mb_in)
    delete_batches(mb_out)
    delete_batches(mb_test_in)
    deallocate_ann(ann)
    
    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics('lineno')
    
    #print(top_stats)

