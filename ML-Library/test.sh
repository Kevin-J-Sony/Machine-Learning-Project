#!/bin/bash

if [ -e test.out ]
then
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
    ./test.out
else
    echo "test.out does not exist. Run 'make test'"
fi

