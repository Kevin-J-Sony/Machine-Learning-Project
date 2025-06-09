CC=gcc
LINK=gcc

# Calling 'make' should invoke 'make library'
all: library


test: test.c libmymllib.so
	$(CC) 

library: matrix.o
	gcc -shared -o libmymllib.so matrix.o

matrix.o: src/math/matrix.c
	$(CC) -Wall -fPIC -c src/math/matrix.c -o matrix.o

clean:
	rm libmymllib.so matrix.o