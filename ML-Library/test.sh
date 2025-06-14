#!/bin/bash

if [ -e test.out ]
then
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
    # gdb ./test.out core
    valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./test.out 2> stderr.txt
    # ./test.out
else
    echo "test.out does not exist. Run 'make test'"
fi

