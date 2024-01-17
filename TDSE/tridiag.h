#ifndef TRIDIAG_H
#define TRIDIAG_H

void Inv_Tridiagonal_Matrix_complex_Numerov( double *, double *, double *, double *, double*, int );
int QL_Tridiagonal_Symmetric_Matrix( double *, double *,double *, int, int);
double Calculate_Shift(double, double, double); 
void Transform_Matrix(double*, double, double, int, int);
void Inv_Tridiagonal_Matrix_complex( double *, double *, double *, double *, double *, int );
void Inv_Tridiagonal_Matrix( double *, double *, double *, double *, double *, int);
double Einitialise(double *,double *,double *,double *,double *,double,double ,int);
double E_calculation(double *,double *,double *,double *,int);
double E_calculation_numerov(double *,double ,double *,int );

#endif