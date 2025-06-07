CC=gcc
LINK=gcc

test: test.o matrix.o
	$(LINK) test.o matrix.o -o test_library

test.o: test/test.c
	$(CC) -c test/test.c -o test.o

mllib.o: src/mllib.c
	$(CC) -c src/mllib.c -o mllib.o

matrix.o: src/math/matrix.c
	$(CC) -c src/math/matrix.c -o matrix.o

clean:
	rm mllib.o test.o test_library
