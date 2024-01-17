#ifndef TOOLS_H
#define TOOLS_H

//#include "/usr/local/include/fftw3.h"
#include <complex.h>
#include <fftw3.h>
#include "../HDF5_tools/h5_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "util.h"


void solve_UPPE(char * init_file, int steps, int source_term, double w_max, const char * dir);
void RK4_step(hid_t * file_id, int source_term, int step, double w_max, const char * dir);
herr_t TDSE_execution(hid_t *file_id);
double * omegas(int N, double dt);
double sigmoid(double x, double x0, int p, double omega_0);
double super_gaussian(double x, double x0, double sigma, int n);
double bandpass(double x, double x_min, double x_max, double xc, int p, int n, double omega_0);
double highpass(double x, double xc, double x0, int n, double omega_0);
double n(double omega, double density, double omega_0);
double d_n(double omega, double density, double omega_0);
double d_n_acc(double omega, double density, double omega_0);
double n_acc(double omega, double density, double omega_0);
double chi3(double omega, double omega_0);
void complex_power_3(fftw_complex * E, fftw_complex * out, int size);
void temporal_filter(fftw_complex *in, double * time, int size);
void phase_shift(fftw_complex * in_out, double * w, int N, double T, int sign);
void normalize_field(fftw_complex * in_out, int N);
void dipole_acceleration_filtered(fftw_complex * P_acc, 
                                  fftw_complex * grad_V,
                                  fftw_complex * E,
                                  fftw_complex * t_filter,
                                  int N);

#endif