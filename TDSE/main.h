#ifndef MAIN_H
#define MAIN_H

void tqli(double *,double *, int n,double *);
void Initialise(int);
void gaussian(void);
void volkov_state(void);
void volkov_state_vg(void);
double* rmv_gs(double*,double*,double*,double);
double potential(double);
double gradpot(double);

#endif