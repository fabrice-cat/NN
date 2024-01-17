#ifndef PROP_H
#define PROP_H

#include "../HDF5_tools/h5_tools.h"

double norme(double *,int);
void normalise(double *,int);
double* propagation(double,double,double,int,int,int,double,int,int,double,double *,double *,double *
				 ,hid_t *,herr_t *,double,double,double*,double*);

double EField(double,double,double,double,int,double,double);
double AField(double,double,double,double,int,double,double);
double* extend_grid(double *,int,int,int);
void window_analysis(double,double,double,int,int,double,double*,double*,double*,double*,double*);
void dipole_analysis(double,double,double*,double*,int,int);

#endif