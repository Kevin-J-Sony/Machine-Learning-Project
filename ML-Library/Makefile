CC=gcc
LINK=gcc

# Calling 'make' should invoke 'make library'
all: library


test: library
	$(CC) test/test.c -L. -lmymllib -o test.out
	
library: matrix.o batch.o
	gcc -shared -o libmymllib.so matrix.o batch.o

matrix.o: src/math/matrix.c
	$(CC) -Wall -fPIC -c src/math/matrix.c -o matrix.o

batch.o: src/processing/batch.c
	$(CC) -Wall -fPIC -c src/processing/batch.c -o batch.o

clean:
	rm libmymllib.so *.o test.out