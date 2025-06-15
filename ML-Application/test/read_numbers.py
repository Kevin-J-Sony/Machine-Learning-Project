import numpy as np

def read_train_data(number_of_data):
    file = open("../data/train-images.idx3-ubyte", 'rb')
    a = file.read(4)
    b = file.read(4)
    length = file.read(4)
    width = file.read(4)

    raw_numbers = []
    for iter in range(number_of_data):
        raw_number = [float(0) for i in range(28 * 28)]
        for i in range(28 * 28):
            a = int.from_bytes(file.read(1), "big")
            raw_number[i] = a / 256
        raw_numbers.append(raw_number)

    file.close

    file = open("../data/train-labels.idx1-ubyte", 'rb')
    a = file.read(4)
    b = file.read(4)

    labels = []
    for label in range(number_of_data):
        l = [0 for i in range(10)]
        l[int.from_bytes(file.read(1), "big")] = 1
        labels.append(l)
    
    file.close()

    return raw_numbers, labels


def read_test_data(number_of_data):
    file = open("../data/t10k-images.idx3-ubyte", 'rb')
    a = file.read(4)
    b = file.read(4)
    length = file.read(4)
    width = file.read(4)

    raw_numbers = []
    for iter in range(number_of_data):
        raw_number = [float(0) for i in range(28 * 28)]
        for i in range(28 * 28):
            a = int.from_bytes(file.read(1), "big")
            raw_number[i] = a / 256
        raw_numbers.append(raw_number)

    file.close

    file = open("../data/t10k-labels.idx1-ubyte", 'rb')
    a = file.read(4)
    b = file.read(4)

    labels = []
    for label in range(number_of_data):
        l = [0 for i in range(10)]
        l[int.from_bytes(file.read(1), "big")] = 1
        labels.append(l)

    file.close()

    return raw_numbers, labels


if __name__ == "__main__":
    train_input, train_label = read_test_data(1)
    for row in range(28):
        s = []
        for col in range(28):
            s.append(train_input[0][row * 28 + col])
        print(s)
    print("\n\n\n")
    print(train_label)