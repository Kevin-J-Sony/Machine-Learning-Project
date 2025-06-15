CC=gcc
LINK=gcc

# Calling 'make' should invoke 'make library'
all: library


test: library
	$(CC) test/test.c -L. -lmymllib -g -o test.out
	
library: matrix.o batch.o ann.o
	gcc -shared -o libmymllib.so matrix.o batch.o ann.o

static_library: matrix.o batch.o ann.o
	ar rcs staticmllib.a matrix.o batch.o ann.o

matrix.o: src/math/matrix.c
	$(CC) -Wall -g -fPIC -c src/math/matrix.c -o matrix.o

batch.o: src/processing/batch.c
	$(CC) -Wall -g -fPIC -c src/processing/batch.c -o batch.o

ann.o: src/unsupervised/ann.c
	$(CC) -Wall -g -fPIC -c src/unsupervised/ann.c -o ann.o


clean:
	rm libmymllib.so *.o test.out