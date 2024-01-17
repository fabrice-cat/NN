#include "util.h"

/**
 * @brief Save array into a separate binary file
 * 
 * @param arr 1D array pointer.
 * @param size Size of the array.
 * @param filename Name of the binary file including suffix.
 */
void export_arr(double *arr, int size, char * filename) {
    FILE * f;
    f = fopen(filename, "w");
    for (int i = 0; i < size; i++) {
        fprintf(f, "%e \n", arr[i]);
    }
    fclose(f);
}

/**
 * @brief Creates file name in each step in a given directory.
 * 
 * @param filename Result file name.
 * @param dir File directory.
 * @param name File name, e.g. "data_".
 * @param step Step of the UPPA iteration. 
 */
void make_filename(char * filename, const char * dir, char * name, int step) {
    char integer_string[10];
    sprintf(integer_string, "%d", step);
    strcpy(filename, dir);
    strcat(filename, name);
    strcat(filename, integer_string);
    strcat(filename, ".h5");
}