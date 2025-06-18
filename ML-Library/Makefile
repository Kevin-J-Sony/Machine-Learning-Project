CC=gcc
LINK=gcc

# Calling 'make' should invoke 'make library'
all: library


test: library
	$(CC) test/test.c -L. -lmymllib -g -o test.out

library: matrix.o batch.o ann.o
	gcc -g -shared -o libmymllib.so matrix.o batch.o ann.o

static_library: matrix.o batch.o ann.o
	ar rcs staticmllib.a matrix.o batch.o ann.o

matrix.o:
	$(CC) -Wall -g -fPIC -c src/math/matrix.c -o matrix.o

batch.o:
	$(CC) -Wall -g -fPIC -c src/processing/batch.c -o batch.o

ann.o:
	$(CC) -Wall -g -fPIC -c src/neural_network/ann.c -o ann.o


clean:
	rm libmymllib.so *.o test.out


test_for_mac:
	clang -fsanitize=address -g -O1 -Wall test/test.c src/math/matrix.c src/processing/batch.c src/neural_network/ann.c -o test_mac_exec
