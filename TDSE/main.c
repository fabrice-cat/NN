#include<time.h>
#include<stdio.h>
#include<stdlib.h>
//#include<malloc.h>
#include<math.h>
#include "prop.h"
#include "tridiag.h"
#include "../HDF5_tools/h5_tools.h"
#include "main.h"


double *y,*x,*a,*c,sum,*diagonal,*off_diagonal,*eigenvector,*u,*r,*vector;
double *timet,*dipole;
double dx,dt,xmax,Eguess,Einit,CV,phi,omega,E0,period,Pi,tfinal,alpha,v,mod1,mod2,dE,Estep,norm_gauss;
double *psi0,*psi,*psi2,Einit2,ps_re,ps_im,*psiexc,*psi_rmv_gs;
double E_start,ton,toff,dw;
int num_E,num_exp,num_w;
int N_total;


double *test_expand;

double *dinf,*dsup,*d,*u1,*res,*t;

int i,j,k,l,m,num_r,num_t,err,max_iteration_count,size,nc;
clock_t start, finish;

double *pot, *Field;

int size_exp,shift;

//FILE *eingenvaluef,*eingenvectorf,*timef,*gaussianwp,*volkovwp,*param,*pot;

// HDF5 file
hid_t   file_id, dataset_id, dataspace_id; /* identifiers */
hsize_t dims_2d[2], dims_1d[1];
herr_t  status;
/*
herr_t read_h5_data(hid_t file_id, const char * dset, hid_t Datatype, void * buf) {
    hid_t dataset_id;
    herr_t status;
    dataset_id = H5Dopen2(file_id, dset, H5P_DEFAULT);
	status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
	status = H5Dclose(dataset_id);
    return status;
}
*/

int main(int argc, char **argv)
{
	Pi = acos(-1.);

	// Open the param.txt file for intialisation of the parameter
	/*
	param = fopen("param.txt" , "r");
	
	if(param == NULL) {printf("DATA could not be found in param.txt file\n");}
	
	
	fscanf(param,"%lf",&Eguess); // Energy of the initial state
	fscanf(param,"%i",&num_r); // Number of points of the initial spatial grid
	fscanf(param,"%i",&num_exp); // Number of points of the spatial grid for the expansion
	fscanf(param,"%lf",&dx); // resolution fo the grid
	fscanf(param,"%lf",&E0); // Amplitude of the Electric field
	fscanf(param,"%lf",&omega); // Frequency of the laser
	fscanf(param,"%lf",&phi); // Phase of the CEP in 0.5*Pi unit
	fscanf(param,"%lf",&ton); // Turn on 
	fscanf(param,"%lf",&toff); // Turn off 
	fscanf(param,"%i",&nc); // number of cycle
	fscanf(param,"%i",&num_t); // number of points per cycle
	fscanf(param,"%lf",&dE); // resolutiomn in energy
	fscanf(param,"%lf",&Estep); // Step for the energy analysis
	fscanf(param,"%i",&num_E); // Number of points of the electron spectrum
	fscanf(param,"%lf",&E_start); // Starting energy of the electron spectrum
	fscanf(param,"%i",&num_w); // Number of points of the HHG spectrum
	fscanf(param,"%lf",&dw); // Step for the HHG spectra
	*/
	
	file_id = H5Fopen(argv[1], H5F_ACC_RDWR, H5P_DEFAULT);
	//group_id = H5Gopen2(file_id, "/inputs", H5P_DEFAULT);
	/* 
	dataset_id = H5Dopen2(file_id, "/inputs/Eguess", H5P_DEFAULT);
	status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Eguess);
	status = H5Dclose(dataset_id);
	*/
	status = read_h5_data(file_id, "/inputs/Eguess", H5T_NATIVE_DOUBLE, &Eguess);
	status = read_h5_data(file_id, "/inputs/num_r", H5T_NATIVE_INT32, &num_r);
	status = read_h5_data(file_id, "/inputs/num_exp", H5T_NATIVE_INT32, &num_exp);
	status = read_h5_data(file_id, "/inputs/dx", H5T_NATIVE_DOUBLE, &dx);
	status = read_h5_data(file_id, "/inputs/E0", H5T_NATIVE_DOUBLE, &E0);
	status = read_h5_data(file_id, "/inputs/omega", H5T_NATIVE_DOUBLE, &omega);
	status = read_h5_data(file_id, "/inputs/phi", H5T_NATIVE_DOUBLE, &phi);
	status = read_h5_data(file_id, "/inputs/ton", H5T_NATIVE_DOUBLE, &ton);
	status = read_h5_data(file_id, "/inputs/toff", H5T_NATIVE_DOUBLE, &toff);
	status = read_h5_data(file_id, "/inputs/nc", H5T_NATIVE_INT32, &nc);
	status = read_h5_data(file_id, "/inputs/num_t", H5T_NATIVE_INT32, &num_t);
	status = read_h5_data(file_id, "/inputs/dE", H5T_NATIVE_DOUBLE, &dE);
	status = read_h5_data(file_id, "/inputs/Estep", H5T_NATIVE_DOUBLE, &Estep);
	status = read_h5_data(file_id, "/inputs/num_E", H5T_NATIVE_INT32, &num_E);
	status = read_h5_data(file_id, "/inputs/E_start", H5T_NATIVE_DOUBLE, &E_start);
	status = read_h5_data(file_id, "/inputs/num_w", H5T_NATIVE_INT32, &num_w);
	status = read_h5_data(file_id, "/inputs/dw", H5T_NATIVE_DOUBLE, &dw);
	status = read_h5_data(file_id, "/inputs/N_total", H5T_NATIVE_INT32, &N_total);
	

	// NOTE: arrays in Fortran are column major, in C are row major -> for compatibility
	// we stick to column major for fields, dipoles etc. in h5 file
	//double Field[nc*num_t+1][2];
	//status = read_h5_data(file_id, "/outputs/Efield", H5T_NATIVE_DOUBLE, &Field);

	
	/*
	for (int i = 0; i < 10; i++) {
		printf("Field value: %lf \n", Field[i][0]);
		printf("Field value: %lf \n", Field[i][1]);
	}
	*/
	
	/*printf("Field value: %lf \n", Field[1][0]);
	printf("Field value: %lf \n", Field[1][1]);
	printf("Field value: %lf \n", Field[1][2]);
	printf("Field value: %lf \n", Field[1][3]);
	printf("\n");
*/
	//status = H5Gclose(group_id);

	printf("\n");
	printf("Parameters of the calculation for TDSE in 1D in the velocity GAUGE (c) DiMauro group OSU\n");
	printf("\n");
	printf("Number of points of the initial spatial grid : %i \n",num_r);
	printf("Number of points of the expansion of the grid : %i \n",num_exp);
	printf("Resolution of the spatial grid : %lf \n",dx);
	printf("Amplitude of the field : %lf (a.u.) \n",E0);
	printf("Frequency of the laser : %lf (a.u.)\n",omega);
	printf("Phase of the CEP for sin sqare pulse : %lf (in 0.5*Pi unit)\n",phi);
	printf("Turn on : %lf (in period unit)\n",ton);
	printf("Turn off: %lf (in period unit)\n",toff);
	printf("Number of cycles of the pulse : %i \n",nc);
	printf("Number of points per cycle : %i \n",num_t);
	printf("Resolution in energy for the electron spectrum : %lf \n",dE);
	printf("Step for the energy : %lf \n",Estep);
	printf("Number of points of the electron spectrum : %i \n",num_E);
	printf("Starting energy of the electron spectrum : %lf \n",E_start);
	printf("Number of points of the HHG spectrum : %i \n",num_w);
	printf("Step for the HHG spectra : %lf \n",dw);
	printf("Number of points for the computation total : %i \n",N_total);
	printf("\n");

	// for the test 
//	num_r = 5; dx = 0.1; dt = 2;
	// ....

	//return 0;


	size = 2*(num_r+1);// for complex number


	// Allocation memory
	x = calloc((num_r+1),sizeof(double)); 
	off_diagonal = calloc(size,sizeof(double));
	diagonal = calloc(size,sizeof(double));
	vector = calloc(size,sizeof(double));
	psi0 = calloc(size,sizeof(double));
	psi2 = calloc(size,sizeof(double));
	psi = calloc(size,sizeof(double));
	psiexc = calloc(size,sizeof(double));
	t = calloc(N_total,sizeof(double));
	dipole = calloc(2*N_total,sizeof(double));

	status = read_h5_data(file_id, "/outputs/tgrid", H5T_NATIVE_DOUBLE, t);

	period = 2.*Pi/omega;
	dt = t[1]-t[0];
	phi = phi*Pi*0.5;

	pot = calloc((num_r+1), sizeof(double));
	//Field = calloc((num_r+1), sizeof(double));


	/*
	eingenvaluef = fopen("eingenvalue.dat", "w" );
	eingenvectorf = fopen("eingenvector.dat", "w" );
	pot = fopen("latice.dat", "w" );

	if(eingenvaluef==NULL || eingenvectorf==NULL)
	{printf("pb d'ouverture de fichier");}
	*/
	

	printf("Initialisation \n");

	// Initialise vectors and Matrix 
	Initialise(num_r);
	for(i=0;i<=num_r;i++){
		psi0[2*i] = 1.0; 
		psi0[2*i+1] = 0.; 
		psiexc[2*i] = 1; 
		psiexc[2*i+1] = 0.; 	
	}
	//normalise(psi0,num_r); // Initialise psi0 for Einitialise
	//normalise(psiexc,num_r);

	CV = 1E-15; // CV criteria

	/* This number has to be small enough to assure a good convregence of the wavefunction
	if it is not the case, then the saclar product of the the ground state and the excited states 
	is not quite 0 and those excited appears in the energy analysis of the gorund states, so the propagation !!
	CV = 1E-25 has been choosen to have a scalar product of 10^-31 with the third excited state for num_r = 5000 and dx=0.1
	*/

	printf("Calculation of the energy of the ground sate ; Eguess : %f\n",Eguess);

	Einit = Einitialise(psi0,off_diagonal,diagonal,off_diagonal,x,Eguess,CV,num_r);
	
	//for(i=0;i<=num_r;i++) {fprintf(eingenvectorf,"%f\t%e\t%e\n",x[i],psi0[2*i],psi0[2*i+1]); fprintf(pot,"%f\t%e\n",x[i],potential(x[i]));}
	for (int i = 0; i<=num_r; i++) {
		pot[i] = potential(x[i]);
	}

	dims_1d[0] = num_r+1;
	//dims[1] = 1;
	status = create_dataset_array(file_id, "/outputs/xgrid", H5T_NATIVE_DOUBLE, 1, dims_1d);
	status = write_to_h5(file_id, "/outputs/xgrid", H5T_NATIVE_DOUBLE, x);
	status = create_dataset_array(file_id, "/outputs/potential", H5T_NATIVE_DOUBLE, 1, dims_1d);
	status = write_to_h5(file_id, "/outputs/potential", H5T_NATIVE_DOUBLE, pot);
	dims_1d[0] = 2*(num_r+1);
	status = create_dataset_array(file_id, "/outputs/ground_state", H5T_NATIVE_DOUBLE, 1, dims_1d);
	status = write_to_h5(file_id, "/outputs/ground_state", H5T_NATIVE_DOUBLE, psi0);
	status = create_dataset_scalar(file_id, "/outputs/E_ground_state", H5T_NATIVE_DOUBLE);
	status = write_to_h5(file_id, "/outputs/E_ground_state", H5T_NATIVE_DOUBLE, &Einit);
	 
	printf("Initial energy is : %1.12f\n",Einit);
	printf("first excited energy is : %1.12f\n",Einit2);


	printf("\n");	
	printf("Propagation procedure ...\n");
	printf("\n");	

	start = clock();
	
//	Initialise(5);
/*
	nc = 1;
	num_t = 1500;
	dt=0.01;
    num_exp = 0;
	tfinal = (nc*num_t)*dt;
*/


	//projection_analysis_EV(dE,Estep,E_start,num_E,num_r,dx,psi0,off_diagonal,diagonal,off_diagonal,x);


	psi = propagation(E0,omega,phi,nc,N_total,num_t,dt,num_r,num_exp,dx,psi0,psi,x,&file_id,&status,ton,toff,t,dipole);

//	volkov_state_vg();

	/* REMOVING PHOTOELECTRON SPECTRUM COMPUTATION

	printf("\n");
	printf("Calculation of the HHG spectrum \n");

	//dipole_analysis(num_w,dw,timet,dipole,nc,num_t);


	printf("\n");
	printf("Calculation of the electron spectrum \n");

	
	num_r = num_r + (nc-1)*num_exp; // the fisrt cycle does not expand the grid !

	Initialise(num_r);
	psi0 = calloc(2*(num_r+1),sizeof(double)); // allocate psi0 with the right final size



	for(i=0;i<=num_r;i++){psi0[2*i] = 1; psi0[2*i+1] = 0.;}
	Einit = Einitialise(psi0,off_diagonal,diagonal,off_diagonal,x,Einit,CV,num_r); // Psi0 with the new num_r
	printf("Initial energy is : %1.12f\n",Einit);
	//for(i=0;i<=num_r;i++) {fprintf(eingenvectorf,"%f\t%e\t%e\n",x[i],psi0[2*i],psi0[2*i+1]); fprintf(pot,"%f\t%e\n",x[i],potential(x[i]));}


	// calculation of the electron spectrum
//	for(i=0;i<=num_r;i++) fprintf(pot,"%f\t%e\n",x[i],potential(x[i]));

	// This is to establish the electron spectrum

	//free(psi);
	//psi = calloc(2*(num_r+1),sizeof(double)); // allocate psi0 witn the right fianl size
	//for(i=0;i<=num_r;i++){psi[2*i] = 1.0+x[i]; psi[2*i+1] = 0.0;} 


	// Remove the Ground state from Psi
	psi_rmv_gs = calloc(2*(num_r+1),sizeof(double));
	psi_rmv_gs = rmv_gs(psi0,psi,x,num_r);




	window_analysis(dE,Estep,E_start,num_E,num_r,dx,psi,off_diagonal,diagonal,off_diagonal,x);
	//projection_analysis_EV(dE,Estep,E_start,num_E,num_r,dx,psi,off_diagonal,diagonal,off_diagonal,x);

	*/
	finish = clock();


	
	//fclose(eingenvectorf);


	printf("\n");
	printf("Duration of calculation %f sec\n",(double)(finish - start) / CLOCKS_PER_SEC);
	printf("\n");
	
	printf("Calculation terminated ; good analysis\n");


	status = H5Fclose(file_id);
	free(psi); free(psi0); free(x); free(off_diagonal);
	free(diagonal); free(vector); free(psi2); free(psiexc); 
	free(t); free(dipole); 
	
	return 0;

/*
*/

}



void Initialise(int num_r)
{
        double xmax = 0.5*num_r*dx;
	x = calloc((num_r+1),sizeof(double));
	off_diagonal = calloc(2*(num_r+1),sizeof(double));
	diagonal = calloc(2*(num_r+1),sizeof(double));	

	//Initialisation Matrix corresponding to D2
	for(i=0;i<=num_r;i++)
	{
		x[i] = i*dx-xmax; 
		off_diagonal[2*i] = -0.5/(dx*dx); off_diagonal[2*i + 1] = 0.;
		diagonal[2*i] = 1./(dx*dx); diagonal[2*i + 1] = 0.;
	}

	
}
/*



void gaussian(void)
{

 double phi1,phi2,phi3,mod1,mod2,psigaussian_re,psigaussian_im;

 

 gaussianwp = fopen("gaussianwp.dat", "w" );

for(i=0;i<=num_r;i++)
{
  phi1 = v*x[i] - 0.5*v*v*tfinal;
  
  phi2 = 0.125*tfinal*pow(pow(alpha,4)+tfinal*tfinal/4.,-1);
  phi2=phi2*(x[i]-v*tfinal)*(x[i]-v*tfinal);

  phi3 = -0.5*atan(0.5*tfinal/(alpha*alpha));

  mod1 = -0.25*alpha*alpha*pow(pow(alpha,4)+tfinal*tfinal/4.,-1);
  mod1 = mod1*pow(x[i]-v*tfinal,2);
  mod2 = -0.25*log(pow(alpha,4)+tfinal*tfinal/4);

  psigaussian_re = norm_gauss*sqrt(Pi)*exp(mod1+mod2)*cos(phi1+phi2+phi3);
  psigaussian_im = norm_gauss*sqrt(Pi)*exp(mod1+mod2)*sin(phi1+phi2+phi3);

  fprintf(gaussianwp,"%f\t%f\t%f\n",x[i],psigaussian_re,psigaussian_im);
}


 fclose(gaussianwp);

}



void volkov_state(void)
{
 
double phi1,phi2,phi3,phi4,mod1,mod2,psivolkov_re,psivolkov_im;
double At,intA,intA2;
double beta_re,beta_im,num; 


volkovwp = fopen("volkovwp.dat", "w" );


At = E0/omega*(cos(omega*tfinal)-1);
intA = E0/omega*(sin(omega*tfinal)/omega-tfinal);
intA2 = (E0/omega)*(E0/omega)*(1.5*tfinal+0.25*sin(2*omega*tfinal)/omega-2*sin(omega*tfinal)/omega);


for(i=0;i<=num_r;i++)
{

  phi1 = -x[i]*At;
  
  phi2 = -0.5*intA2;

  phi3 = -0.5*atan(0.5*tfinal/(alpha*alpha));

  beta_re = 2*alpha*alpha*v;
  beta_im = x[i]+intA;

  num = pow(alpha,4)+tfinal*tfinal/4.;	

  phi4 = 0.25*(2*alpha*alpha*beta_re*beta_im-0.5*tfinal*(beta_re*beta_re-beta_im*beta_im))/num;

  mod1 = 0.25*(alpha*alpha*(beta_re*beta_re-beta_im*beta_im)+tfinal*beta_im*beta_re)/num;

  mod2 = -0.25*log(num);

  psivolkov_re = norm_gauss*sqrt(Pi)*exp(mod1+mod2)*cos(phi1+phi2+phi3+phi4)*exp(-v*v*alpha*alpha);
  psivolkov_im = norm_gauss*sqrt(Pi)*exp(mod1+mod2)*sin(phi1+phi2+phi3+phi4)*exp(-v*v*alpha*alpha);

  fprintf(volkovwp,"%f\t%f\t%f\n",x[i],psivolkov_re,psivolkov_im);
}


 fclose(volkovwp);

}

void volkov_state_vg(void)
{
 
double phi1,phi2,phi3,mod1,mod2,mod3,psivolkov_re,psivolkov_im,xp;
double intA;
double num; 


volkovwp = fopen("volkovwp_vg.dat", "w" );

printf("TIME FOR VOLKOV : %f \n",tfinal);

intA = E0/omega*(-cos(omega*tfinal)/omega+1./omega);

norm_gauss = pow(2*alpha/Pi,0.25);

for(i=0;i<=num_r;i++)
{
  xp = x[i] + intA;
  num = pow(alpha,4)+tfinal*tfinal/4.;	

  mod1 = (4*pow(alpha,4)*v*v - pow(xp,2.))*alpha*alpha;
  mod1 = 0.25*mod1/num;

  mod2 = 2*xp*alpha*alpha*tfinal*v;
  mod2 = 0.25*mod2/num;

  phi1 = 4*pow(alpha,4)*xp*v;
  phi1 = 0.25*phi1/num;

  phi2 = -0.5*tfinal*(4*pow(alpha,4)*v*v-xp*xp);
  phi2 = 0.25*phi2/num;

  mod3 = -0.25*log(num);	
  phi3 = -0.5*atan(0.5*tfinal/(alpha*alpha));

  psivolkov_re = norm_gauss*sqrt(Pi)*exp(mod1+mod2+mod3)*cos(phi1+phi2+phi3)*exp(-v*v*alpha*alpha);
  psivolkov_im = norm_gauss*sqrt(Pi)*exp(mod1+mod2+mod3)*sin(phi1+phi2+phi3)*exp(-v*v*alpha*alpha);


  psi0[2*i] = psivolkov_re;
  psi0[2*i + 1] = psivolkov_im;


  fprintf(volkovwp,"%f\t%f\t%f\n",x[i],psivolkov_re,psivolkov_im);
}

 fclose(volkovwp);

}
*/


double* rmv_gs(double *psi0,double *psi, double *x, double num_r)
{

  double *psi_new;
  double c_gs_re,c_gs_im;
  int j;

  psi_new = calloc(2*(num_r+1),sizeof(double));  

  c_gs_re = 0.0; c_gs_im = 0.0;
  for(j = 0 ; j<= num_r ; j++) 
  {
     c_gs_re += psi[2*j]*psi0[2*j];
     c_gs_im += psi[2*j+1]*psi0[2*j];
  }

  for(j = 0 ; j<= num_r ; j++) 
  {
     psi_new[2*j] =  psi[2*j] - c_gs_re*psi0[2*j];
     psi_new[2*j+1] =  psi[2*j+1] - c_gs_im*psi0[2*j];
  }

  return psi_new;

  free(psi_new);

}

double potential(double x)
{

	double V,c,V0,l,a;

	//c = 2.0; 
	c = 1/(4*Pi);
	V0 = 1.252;
	l = 1.0;
	//a = 1.414213562; //sqrt(2) -> Hydrogen
	a = 1.1893; // -> Argon
	//a = 1.205;//1.5; //0.695;//*0.25; // 1.205 for Argon -> Eground = -0.57277
	//V = -V0*exp(-l*abs(x))*pow(c+x*x,-0.5);

	//V = -3.0*pow((exp(x)+exp(-x))/2.0,-2.0);


	V = -1.0/sqrt(a*a+x*x);

	// if(sqrt(x*x) < a)
	// {  
	//	V = -0.015;
	// }
	// else
	// {V = 0.0;}
	

	return V;

}

double gradpot(double x)
{
   double c,R,a;
   //c = 0.605;
   c = 1/(4*Pi);
   //R = 1.9;
   //a = 1.205; 
   //a = 1.5;
   //a = 1.414213562; //sqrt(2) -> Hydrogen
   a = 1.1893; // -> Argon
   
  //return x*pow(c*c+(x-R*0.5)*(x-R*0.5),-1.5)+x*pow(c*c+(x+R*0.5)*(x+R*0.5),-1.5);
  return x*pow(a*a+x*x,-1.5);


}


