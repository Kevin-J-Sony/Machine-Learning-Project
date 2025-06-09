
// Self-evidently, there will be no other library by this name, but out of respect for the standard
// maintain this convention. Furthermore, we also ensure that all methods are defined within this "if"
#ifndef ML_LIB_H
#define ML_LIB_H

// Define a simple boolean data type for convenience
typedef int boolean;
#define TRUE (1 == 1)
#define FALSE (1 == 0)

// Define a debug mode to quickly discover bugs
#define ML_LIB_DEBUG_MODE

// Some files to include in debug mode
#ifdef ML_LIB_DEBUG_MODE
#include <stdio.h>
#include <stdlib.h>
#endif


#endif