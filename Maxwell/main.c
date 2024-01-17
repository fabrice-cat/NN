#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../HDF5_tools/h5_tools.h"
#include "tools.h"
#include <fftw3.h>
//#include "/usr/local/include/fftw3.h"
#include "util.h"

#define STD_OUT
#define DEBUG

/**
 * @brief Main
 * 
 * @details Compile using make and run with command line arguments ```argv```.
 * 
 * @param argc No. of arguments.
 * @param argv Arguments for Maxwell code, argv[1] must be an initial file name, 
 * argv[2] the directory, argv[3] is number of iterations of Maxwell, argv[4] is
 * the source term (0 == ab-initio w/ grad V term, 1 == ad-hoc, 2 == ab-initio 
 * w/ P = -<x> term), argv[5] is the harmonic cutoff.
 * @return int 
 */
int main(int argc, char **argv) {
    /* Time parameters */
    clock_t start, finish;

    // Iterations, source term, cutoff
    int iter, src;
    double cutoff;
    iter = atoi(argv[3]);
    src = atoi(argv[4]);
    cutoff = atof(argv[5]);
    printf("Iterations: %d \n", iter);
    printf("Source term: %d \n", src);
    printf("Cutoff: %f \n", cutoff);

    // Start timer
    start = clock();

    /***** MAIN PART OF PROGRAM *****/
    solve_UPPE(argv[1], iter, src, cutoff, argv[2]);

    // End timer
    finish = clock();

    #ifdef STD_OUT

    printf("\n");
	printf("Duration of calculation for Maxwell: %f sec\n", (double)(finish - start) / CLOCKS_PER_SEC);
	printf("\n");

    #endif


    return 0;
}