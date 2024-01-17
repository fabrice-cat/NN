#include<math.h>
//#include<malloc.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include "tridiag.h"
#include "main.h"
#include "../HDF5_tools/h5_tools.h"

#define Pi acos(-1.)

//#pragma warning( disable : 4996 ) // warning for fopen in visual 2005

extern double* timet,dipole;

double norme(double *x,int Num_r)
{
	int i;
	double sum = 0.;
	for(i=0;i<=Num_r;i++){sum = sum + x[2*i]*x[2*i] + x[2*i+1]*x[2*i+1];}
	return sum;
}
void normalise(double *x,int Num_r)
{
	int i;
	double sum = 0.;
	for(i=0;i<=Num_r;i++){sum = sum + x[2*i]*x[2*i] + x[2*i+1]*x[2*i+1];}
	sum = sqrt(sum);
	for(i=0;i<=Num_r;i++){x[2*i]/=sum;x[2*i+1]/=sum;}
}


double EField(double E0,double omega,double t,double phi,int n, double ton,double toff)
{
	double omegap = omega/((double)n),ts;
	double a;
	

	//FOR SINE SQUARE

	return E0*sin(omegap*t*0.5)*sin(omegap*t*0.5)*sin(omega*t+phi);
	
    //FOR FALT TOP
/*
	if (t < ton*2*Pi/omega)
	{
		a = sqrt(2)*ton*2*Pi/omega;
		return E0*4*t*t*(a*a-t*t)/(a*a*a*a);
	}
	else
	{
       if (t > (n-toff)*2*Pi/omega)
	   {
			ts = n*2*Pi/omega-t;
			a = sqrt(2)*toff*2*Pi/omega;
		    return E0*4*ts*ts*(a*a-ts*ts)/(a*a*a*a);
	   }
	   else
	   {
		 return  E0*cos(omega*(t-ton*2*Pi/omega));
	   }
	}
*/
	
	// FOR SIN
//	return  E0*sin(omega*t);
}


double AField(double E0,double omega,double t,double phi,int n,double ton,double toff)
{
	
	double omegap = omega/((double)n),ts,Tf;
	double a,b;

	//FOR SINE SQUARE
        if (t > (n*2*Pi/omega))
        { 
          return 0*E0/omega;
        }
        else
        {
	  return E0*sin(omegap*t*0.5)*sin(omegap*t*0.5)*sin(omega*t+phi)/omega;
	}
    //FOR FALT TOP

/*	E0 = E0/omega;

	if (t < ton*2*Pi/omega)
	{
		return E0*pow(sin(0.5*Pi*t/(ton*2*Pi/omega)),4.);
	}
	else
	{
       if (t > (n-toff)*2*Pi/omega)
	   {
			ts = (n-toff)*2*Pi/omega;
			Tf = n*2*Pi/omega;
		   return E0*pow(sin(0.5*Pi*(t-Tf)/(ts-Tf)),4.);
	   }
	   else
	   {
		 return  E0*cos(omega*(t-ton*2*Pi/omega));
	   }
	}
*/
	
	// FOR SIN
//	return  E0*sin(omega*t)/omega;

        if (t > (n*2*3.1415/omega))
        { 
          return 0*E0/omega; printf("Field : %f \n",E0*0);
        }
        printf("Field : %f \n",E0*0);

		  

}


double * extend_grid(double *pold,int size,int oldsize,int shift)
{
	int i;
	double *pnew;

	pnew = calloc(2*(size+oldsize+1),sizeof(double));

	if ((shift >= 0) && (size >= shift))
	{

	 for(i=oldsize+1;i<=oldsize+size;i++) {pnew[2*i] = 0.; pnew[2*i+1] = 0.;}

	 for(i=0;i<=oldsize;i++) 
	 {
		pnew[2*shift+2*(oldsize-i)] = pold[2*(oldsize-i)]; 
		pnew[2*shift+2*(oldsize-i)+1] = pold[2*(oldsize-i)+1]; 
	 }

	for(i=0;i<shift;i++) {pnew[2*i] = 0.; pnew[2*i+1] = 0.;}

	}
	else
	{printf("\nExpansion of the grid incorrect : shift <0 or size < shift \n\n");}
	
	free(pold);

	return pnew;
}

double* propagation(double E0,double omega,double phi,int nc, int N_total,int num_t,double dt,int num_r
,int num_exp,double dx,double *psi0,double *psi,double *x
,hid_t *file_id,herr_t *status, double ton,double toff, double *timet, double *dipole)
{	
	clock_t start, finish;
	double *res1,*dnew1,*dinfnew1,*dsupnew1,*psi_inter1;
	double *res2,*dnew2,*dinfnew2,*dsupnew2,*psi_inter2;
	double Field,tt,coef,pop_re,pop_im,Apot,atten,phiabs;
	int i,j,save,comp,index,k,ip,shift;	
	//FILE *wf,*zplot,*phase;
	char name[20];
	double cpot,c0re,c0im;
	double dip_re,dip_im, grad_pot_re, grad_pot_im;
    double *current, *population, d[N_total][2], Field_total[N_total][2]; 
	double g_pot[N_total][2];
	double *grad_pot;
	double av_x_re,av_px_re,av_x_im,av_px_im,derpsi_re,derpsi_im;

	for (int i = 0; i < N_total; i++) {
		d[i][0] = 0.0;
		d[i][1] = 0.0;
	}
	
	hsize_t dims[1];
	hsize_t dims2D[2];

	psi_inter1 = calloc(2*(num_r+1),sizeof(double));
	res1 = calloc(2*(num_r+1),sizeof(double));
	dnew1 = calloc(2*(num_r+1),sizeof(double));
	dinfnew1 = calloc(2*(num_r+1),sizeof(double));
	dsupnew1 = calloc(2*(num_r+1),sizeof(double));
	psi_inter2 = calloc(2*(num_r+1),sizeof(double));
	res2 = calloc(2*(num_r+1),sizeof(double));
	dnew2 = calloc(2*(num_r+1),sizeof(double));
	dinfnew2 = calloc(2*(num_r+1),sizeof(double));
	dsupnew2 = calloc(2*(num_r+1),sizeof(double));

	current = calloc(N_total, sizeof(double));
	population = calloc(N_total, sizeof(double));
	grad_pot = calloc(2*N_total, sizeof(double));


	*status = read_h5_data(*file_id, "/outputs/Efield", H5T_NATIVE_DOUBLE, Field_total);

	//timef = fopen( "time.dat", "w" );

	//wf = fopen("wf.dat", "w" );

	//phase = fopen("phase.dat", "w" );

	tt = 0;

	//comp = 100; index = 0; save = 1;


	//Field = -EField(E0,omega,tt,phi,nc,ton,toff);
	//Apot = AField(E0,omega,tt,phi,nc,ton,toff);
	//fprintf(timef,"%f\t%f\t%f\t%f\t%f\t%f\t%f\n",tt,-Field,Apot,1.0,0.0,0.0,0.0);

	// Write initial data
	dipole[0] = 0.0;
	dipole[1] = 0.0;
	population[0] = 1.0;
	current[0] = 0.0;
	grad_pot[0] = 0.0;
	grad_pot[1] = 0.0;
	
	
	ip = 0;




	// For the test 
//	num_t = 1; nc = 1;
	// .....


	cpot = 1.;



	for(j = 0 ; j<= num_r ; j++) {psi[2*j] = psi0[2*j]; psi[2*j+1] = psi0[2*j+1];}
	
	start = clock();

	/*
	// Make the expansion of the grid
	if(k != 0) // don't do the expansion for the fisrt cycle
	{
	  shift = num_exp >> 1;
	  psi=extend_grid(psi,num_exp,num_r,shift); psi0=extend_grid(psi0,num_exp,num_r,shift);


	  num_r = num_r + num_exp;

		free(x);
		x = calloc(num_r+1,sizeof(double));
		for(i=0;i<=num_r;i++)
		{
			x[i] = i*dx-0.5*num_r*dx; 
		}

		free(psi_inter1);free(res1);free(dnew1);free(dinfnew1);free(dsupnew1);
		free(psi_inter2);free(res2);free(dnew2);free(dinfnew2);free(dsupnew2);
	
		psi_inter1 = calloc(2*(num_r+1),sizeof(double));
		res1 = calloc(2*(num_r+1),sizeof(double));
		dnew1 = calloc(2*(num_r+1),sizeof(double));
		dinfnew1 = calloc(2*(num_r+1),sizeof(double));
		dsupnew1 = calloc(2*(num_r+1),sizeof(double));

		psi_inter2 = calloc(2*(num_r+1),sizeof(double));
		res2 = calloc(2*(num_r+1),sizeof(double));
		dnew2 = calloc(2*(num_r+1),sizeof(double));
		dinfnew2 = calloc(2*(num_r+1),sizeof(double));
		dsupnew2 = calloc(2*(num_r+1),sizeof(double));

	}

	
	printf("Cycle number : %i ; size of the box : %i \n",k+1,num_r);
	*/


	for(i = 1 ; i< N_total ; i++)
	{
		//tt = tt + dt;
		tt = timet[i];
		
		coef = 0.5*dt/(dx*dx);
		Field = Field_total[i][0];
		//Field = -EField(E0,omega,tt,phi,nc,ton,toff);
		//Field = -(AField(E0,omega,tt+dt,phi,nc,ton,toff)-AField(E0,omega,tt,phi,nc,ton,toff))/dt;
		

		//Apot = AField(E0,omega,tt,phi,nc,ton,toff);

		// element of matrix that will be inverted 

		for(j = 0 ; j<= num_r ; j++) 
		{	
			dinfnew1[2*j] = 1/12.; dinfnew1[2*j+1] = 0.5*dt*( -0.5/(dx*dx) )+0.5*dt*1/12.*(cpot*potential(x[j]));
			dnew1[2*j] = 10/12.; dnew1[2*j+1] = 0.5*dt*( 1./(dx*dx) )+0.5*dt*10/12.*(cpot*potential(x[j]));
			dsupnew1[2*j] = 1/12.; dsupnew1[2*j+1] = 0.5*dt*( -0.5/(dx*dx) )+0.5*dt*1/12.*(cpot*potential(x[j+1]));
			
			/*
			dinfnew2[2*j] = 1/6.+0.5*dt*Apot*0.5/dx; dinfnew2[2*j+1] = 0;
			dnew2[2*j] = 4/6.; dnew2[2*j+1] = 0;
			dsupnew2[2*j] = 1/6.-0.5*dt*Apot*0.5/dx; dsupnew2[2*j+1] = 0;
			*/
		}
		
		// first part of the evolution (H0+V)

		psi_inter1[0] = (10/12.)*psi[0]+coef*psi[1]+1/12.*psi[2]-0.5*coef*psi[3];
		psi_inter1[0] = psi_inter1[0]+0.5*dt*((10/12.)*psi[1]*(cpot*potential(x[0]))
						+(1/12.)*psi[3]*(cpot*potential(x[1])));

		psi_inter1[1] = (10/12.)*psi[1]-coef*psi[0]+1/12.*psi[3]+0.5*coef*psi[2];	
		psi_inter1[1] = psi_inter1[1]-0.5*dt*((10/12.)*psi[0]*(cpot*potential(x[0]))
						+(1/12.)*psi[2]*(cpot*potential(x[1])));

		for(j = 1 ; j< num_r ; j++)
		{

			psi_inter1[2*j] = (10/12.)*psi[2*j]+coef*psi[2*j+1]+1/12.*psi[2*(j+1)]
							  +1/12.*psi[2*(j-1)]-0.5*coef*(psi[2*(j-1)+1]+psi[2*(j+1)+1]);
			psi_inter1[2*j] = psi_inter1[2*j]+0.5*dt*((10/12.)*psi[2*j+1]*(cpot*potential(x[j]))
							  +(1/12.)*psi[2*(j-1)+1]*(cpot*potential(x[j-1]))
							  +(1/12.)*psi[2*(j+1)+1]*(cpot*potential(x[j+1])));

			psi_inter1[2*j+1] = (10/12.)*psi[2*j+1]-coef*psi[2*j]+1/12.*psi[2*(j+1)+1]
							  +1/12.*psi[2*(j-1)+1]+0.5*coef*(psi[2*(j-1)]+psi[2*(j+1)]);
			psi_inter1[2*j+1] = psi_inter1[2*j+1]-0.5*dt*((10/12.)*psi[2*j]*(cpot*potential(x[j]))
							  +(1/12.)*psi[2*(j-1)]*(cpot*potential(x[j-1]))
							  +(1/12.)*psi[2*(j+1)]*(cpot*potential(x[j+1])));

		}

		psi_inter1[2*num_r] = (10/12.)*psi[2*num_r]+coef*psi[2*num_r+1]+1/12.*psi[2*(num_r-1)]-0.5*coef*psi[2*(num_r-1)+1];
		psi_inter1[2*num_r] = psi_inter1[2*num_r]+0.5*dt*((10/12.)*psi[2*num_r+1]*(cpot*potential(x[num_r]))
							  +(1/12.)*psi[2*(num_r-1)+1]*(cpot*potential(x[num_r-1])));

		psi_inter1[2*num_r+1] = (10/12.)*psi[2*num_r+1]-coef*psi[2*num_r]+1/12.*psi[2*(num_r-1)+1]+0.5*coef*psi[2*(num_r-1)];
		psi_inter1[2*num_r+1] = psi_inter1[2*num_r+1]-0.5*dt*((10/12.)*psi[2*num_r]*(cpot*potential(x[num_r]))
							  +(1/12.)*psi[2*(num_r-1)]*(cpot*potential(x[num_r-1])));


		Inv_Tridiagonal_Matrix_complex(dinfnew1,dnew1,dsupnew1,psi_inter1,res1,num_r+1);


		// second part of the evolution (Hint) - length gauge:
		for(j = 0 ; j<= num_r ; j++) {
			psi[2*j] = cos(-Field*dt*x[j])*res1[2*j]-sin(-Field*dt*x[j])*res1[2*j+1]; 
			psi[2*j+1] = cos(-Field*dt*x[j])*res1[2*j+1]+sin(-Field*dt*x[j])*res1[2*j];
		}

		/*
		psi_inter2[0] = 4/6.*res1[0]+(1/6.+0.5*dt*Apot*0.5/dx)*res1[2];
		psi_inter2[1] = 4/6.*res1[1]+(1/6.+0.5*dt*Apot*0.5/dx)*res1[3];

		for(j = 1 ; j< num_r ; j++)
		{

			psi_inter2[2*j] = 4/6.*res1[2*j] + (1/6. + 0.5*dt*Apot*0.5/dx)*res1[2*(j+1)];
			psi_inter2[2*j] = psi_inter2[2*j] + (1/6. - 0.5*dt*Apot*0.5/dx)*res1[2*(j-1)];
			psi_inter2[2*j+1] = 4/6.*res1[2*j+1] + (1/6. + 0.5*dt*Apot*0.5/dx)*res1[2*(j+1)+1];
			psi_inter2[2*j+1] = psi_inter2[2*j+1] + (1/6. - 0.5*dt*Apot*0.5/dx)*res1[2*(j-1)+1];

		}

		psi_inter2[2*num_r] = 4/6.*res1[2*num_r]+(1/6.-0.5*dt*Apot*0.5/dx)*res1[2*(num_r-1)];
		psi_inter2[2*num_r+1] = 4/6.*res1[2*num_r+1]+(1/6.-0.5*dt*Apot*0.5/dx)*res1[2*(num_r-1)+1];	

		Inv_Tridiagonal_Matrix_complex(dinfnew2,dnew2,dsupnew2,psi_inter2,res2,num_r+1);

		*/

/////// uncomment for TESTING
/*
		printf("\n\n\n\n test \n");
		
		printf("\n");
		printf("potential and position\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f %f\n",potential(x[j]),x[j]);
		}
		printf("\n");


		printf("\n");
		printf("Matrix d1\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",dnew1[2*j],dnew1[2*j+1]);
		}
		printf("\n");


		printf("\n");
		printf("Matrix dinf1\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",dinfnew1[2*j],dinfnew1[2*j+1]);
		}
		printf("\n");

		printf("\n");
		printf("Matrix dsup1\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",dsupnew1[2*j],dsupnew1[2*j+1]);
		}
		printf("\n");


		printf("\n");
		printf("psi0\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",psi[2*j],psi[2*j+1]);
		}
		printf("\n");


		printf("\n");
		printf("psiinter1\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",psi_inter1[2*j],psi_inter1[2*j+1]);
		}
		printf("\n");


		printf("\n");
		printf("res1\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",res1[2*j],res1[2*j+1]);
		}
		printf("\n");



		printf("\n");
		printf("Matrix d2\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",dnew2[2*j],dnew2[2*j+1]);
		}
		printf("\n");


		printf("\n");
		printf("Matrix dinf2\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",dinfnew2[2*j],dinfnew2[2*j+1]);
		}
		printf("\n");

		printf("\n");
		printf("Matrix dsup2\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",dsupnew2[2*j],dsupnew2[2*j+1]);
		}
		printf("\n");

		printf("\n");
		printf("psiinter2\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",psi_inter2[2*j],psi_inter2[2*j+1]);
		}
		printf("\n");


		printf("\n");
		printf("res2\n");
		for(j=0;j<=num_r;j++)
		{
			printf("%f +i%f\n",res2[2*j],res2[2*j+1]);
		}
		printf("\n");


		printf("\n\n test end \n\n\n\n");
*/
///////

		// absorbing wall
		/*
		for(j = 0 ; j<= num_r ; j++) 
		{
			if ( j >= (num_r-300) ) 
			{	phiabs = x[j]-x[num_r-300];
				phiabs *= 1./(x[num_r]-x[num_r-300]);
				atten = 1-phiabs;
				phiabs *= Pi*0.5;
				atten *= cos(phiabs);
			}
			else {atten = 1;}

			if ( j <= 300 ) 
			{	phiabs = x[j]-x[300];
				phiabs *= 1./(x[0]-x[300]);
				atten = 1-phiabs;
				phiabs *= Pi*0.5;
				atten *= cos(phiabs);
			}
			else {if (j < (num_r-300)) atten = 1;}
		*/

	/////   Remove the ground state
/*
			if ( tt >= 2*Pi/omega)
			{
			 
				cpot = 0;
			
			  c0re = 0.; c0im = 0.;
			  for(j=0; j<= num_r ; j++)
			  {
			    c0re += res2[2*j]*psi0[2*j]+res2[2*j+1]*psi0[2*j+1];
			    c0im += res2[2*j+1]*psi0[2*j] - res2[2*j]*psi0[2*j+1];
			  }
			  for(j=0; j<= num_r ; j++)
			  {
			    psi[2*j] = res2[2*j] - (psi0[2*j]*c0re-psi0[2*j+1]*c0im);
				psi[2*j+1] = res2[2*j+1] - (psi0[2*j]*c0im + psi0[2*j+1]*c0re);
			  }
			
			  //normalise(psi,num_r);
			}
*/

/*
			atten = 1.;		
		    psi[2*j] = atten*res2[2*j]; 
			psi[2*j+1] = atten*res2[2*j+1];

		}
*/
		pop_re = 0.; 
		pop_im = 0.;
		for(j = 0 ; j<= num_r ; j++) {
			pop_re = pop_re + psi[2*j]*psi0[2*j] + psi[2*j+1]*psi0[2*j+1]; 
			pop_im = pop_im + psi[2*j]*psi0[2*j+1] - psi[2*j+1]*psi0[2*j];
		}

		population[i] = pop_re*pop_re + pop_im*pop_im;

		//fprintf(timef,"%f\t%f\t%f\t%f\n",tt,-Field,Apot,pop_re*pop_re + pop_im*pop_im);

		// calculation of <-grad(V)> and current
		dip_re = 0.; 
		dip_im = 0.;
		grad_pot_re = 0.;
		grad_pot_im = 0.;
		
                
		for(j = 0 ; j <= num_r ; j++) {
			grad_pot_re = grad_pot_re + (psi[2*j]*psi[2*j] + psi[2*j+1]*psi[2*j+1])*gradpot(x[j]); 
			grad_pot_im = grad_pot_im + (psi[2*j]*psi[2*j+1] - psi[2*j+1]*psi[2*j])*gradpot(x[j]);
			dip_re = dip_re + (psi[2*j]*psi[2*j] + psi[2*j+1]*psi[2*j+1])*(x[j]); 
			dip_im = dip_im + (psi[2*j]*psi[2*j+1] - psi[2*j+1]*psi[2*j])*(x[j]);
		}
		dipole[2*i] = dip_re;
		dipole[2*i+1] = dip_im;

		grad_pot[2*i] = grad_pot_re;
		grad_pot[2*i+1] = grad_pot_im;
                 
        current[i] = 0.; 
		for(j = 1 ; j<= num_r-1 ; j++) {
            current[i] = current[i] + psi[2*j]*(psi[2*(j+1)+1]-psi[2*(j-1)+1]) - psi[2*j+1]*(psi[2*(j+1)]-psi[2*(j-1)]);                 
        }
        current[i] = -current[i]*0.5/dx;


		//timet[k*num_t+i] = tt; dipole[2*(k*num_t+i)] = dip_re; dipole[2*(k*num_t+i)+1] = dip_im; 

		//fprintf(timef,"%e\t%e\t%e\t%e\t%e\t%e\t%e\n",tt,Field,Apot,pop_re*pop_re + pop_im*pop_im,dip_re,dip_im,current);

/*
		if (save == comp)
		{
			sprintf(name,"zplot%i.dat",index); 
			//printf("%i\n",save);
			zplot = fopen(name, "w" );
			
			for(j=0;j<=num_r;j++) fprintf(zplot,"%e\n",psi[2*j]*psi[2*j]+psi[2*j+1]*psi[2*j+1]);
			
			fclose(zplot);
			index++;

			save = 1;
		}
		else
		{save++;}
*/
		


		// calculation of the phase diagram <x> and <px>

		/*
		av_x_re = 0; av_px_re = 0; av_x_im = 0; av_px_im = 0; 
		for(j = 0 ; j< num_r ; j++) 
		{ 
			av_x_re = av_x_re + (psi[2*j]*psi[2*j] + psi[2*j+1]*psi[2*j+1])*x[j]; 
			av_x_im = av_x_im + (psi[2*j+1]*psi[2*j] - psi[2*j]*psi[2*j+1])*x[j];


			derpsi_re = (psi[2*(j+1)]-psi[2*j])/dx;
			derpsi_im = (psi[2*(j+1)+1]-psi[2*j+1])/dx;

			av_px_re = av_px_re + psi[2*j]*derpsi_re + psi[2*j+1]*derpsi_im;
			av_px_im = av_px_im + psi[2*j+1]*derpsi_re - psi[2*j]*derpsi_im;
		}

    		av_px_re = av_px_im;
		av_px_im = -av_px_re;
		// N need of dx because psi is normalised


       fprintf(phase,"%e\t%e\t%e\t%e\t%e\n",tt,av_x_re,av_x_im,av_px_re,av_px_im);
		*/

		if (ip == (int)floor(N_total/20.)) {printf("*/"); fflush(stdout); ip = 0;}
		ip++;


	}
	finish = clock();
	printf("\n Duration of calculation %f sec\n\n",(double)(finish - start) / CLOCKS_PER_SEC);
	

	printf("LOCAL TIME %f\n",tt);

	

	
	//for(i=0;i<=num_r;i++) fprintf(wf,"%i\t%e\t%e\t%e\n",i,x[i],psi[2*i],psi[2*i+1]);

	
	dims2D[0] = N_total;
	dims2D[1] = 2;
	dims[0] = 2*(num_r+1);
	*status = create_dataset_array(*file_id, "/outputs/psi", H5T_NATIVE_DOUBLE, 1, dims);
	*status = write_to_h5(*file_id, "/outputs/psi", H5T_NATIVE_DOUBLE, psi);
	*status = create_dataset_array(*file_id, "/outputs/Afield", H5T_NATIVE_DOUBLE, 2, dims2D);
	*status = write_to_h5(*file_id, "/outputs/Afield", H5T_NATIVE_DOUBLE, d);
	for (int k = 0; k < N_total; k++) {
		d[k][0] = dipole[2*k];
		d[k][1] = dipole[2*k+1];
		g_pot[k][0] = grad_pot[2*k];
		g_pot[k][1] = grad_pot[2*k+1];
	}
	*status = create_dataset_array(*file_id, "/outputs/dipole", H5T_NATIVE_DOUBLE, 2, dims2D);
	*status = write_to_h5(*file_id, "/outputs/dipole", H5T_NATIVE_DOUBLE, d);
	/*for (int k = 0; k < N_total; k++) {
		d[k][0] = grad_pot[2*k];
		d[k][1] = grad_pot[2*k+1];
	}*/
	*status = create_dataset_array(*file_id, "/outputs/grad_pot", H5T_NATIVE_DOUBLE, 2, dims2D);
	*status = write_to_h5(*file_id, "/outputs/grad_pot", H5T_NATIVE_DOUBLE, g_pot);
	dims[0] = N_total;
	*status = create_dataset_array(*file_id, "/outputs/current", H5T_NATIVE_DOUBLE, 1, dims);
	*status = write_to_h5(*file_id, "/outputs/current", H5T_NATIVE_DOUBLE, current);
	*status = create_dataset_array(*file_id, "/outputs/population", H5T_NATIVE_DOUBLE, 1, dims);
	*status = write_to_h5(*file_id, "/outputs/population", H5T_NATIVE_DOUBLE, population);

	//fclose(timef); fclose(wf); 
	free(psi_inter1);free(res1);free(dnew1);free(dinfnew1);free(dsupnew1);
    free(psi_inter2);free(res2);free(dnew2);free(dinfnew2);free(dsupnew2);
	free(current);free(population);free(grad_pot);

	return psi;

}


void window_analysis(double dE,double Estep,double E_start,int num_E,int num_r,double dx,double *psi,double *dinf,double *d,double *dsup,double *x)
{

	
	double *dnew,*dnew2,*dinfnew,*dsupnew,*res,*res2,*psi2;
	double *dinfnew2,*dsupnew2;
	double prob;
	int i,j;
	FILE *fel; 


	dnew = calloc(2*(num_r+1),sizeof(double));
	dnew2 = calloc(2*(num_r+1),sizeof(double)); 
	dinfnew = calloc(2*(num_r+1),sizeof(double)); 
	dsupnew = calloc(2*(num_r+1),sizeof(double));
	dinfnew2 = calloc(2*(num_r+1),sizeof(double));
	dsupnew2 = calloc(2*(num_r+1),sizeof(double));
	res = calloc(2*(num_r+1),sizeof(double));
	res2 = calloc(2*(num_r+1),sizeof(double));
	psi2 = calloc(2*(num_r+1),sizeof(double));



	fel = fopen("electron_spectrum.dat","w");
	if(fel == NULL){ printf("Cannot open electron_spectrum.dat"); exit(1);} 
	printf("Working on the bin 00000");



	
/*
	for(j = 0 ; j<= num_r ; j++) 
	{	
			dinfnew[2*j] = -dinf[2*j]; dinfnew[2*j+1] = -dinf[2*j+1];
			dsupnew[2*j] = -dsup[2*j]; dsupnew[2*j+1] = -dsup[2*j+1];
	}
*/





	for(i=0; i<= num_E ; i++)
	{
	 		
	  for(j = 0 ; j<= num_r ; j++) 
	  {	
			//dnew[2*j] = E_start+Estep*i-d[2*j]-potential(x[j])-dE/sqrt(8); dnew[2*j+1] = -d[2*j+1]-dE/sqrt(8);
			//dnew2[2*j] = E_start+Estep*i-d[2*j]-potential(x[j])+dE/sqrt(8); dnew2[2*j+1] = -d[2*j+1]+dE/sqrt(8);

		  dnew[2*j] = 10*(E_start+Estep*i-dE/sqrt(8)-potential(x[j]))/12. - d[2*j]; dnew[2*j+1] = -d[2*j+1]-10*dE/(12*sqrt(8));
		  dnew2[2*j] = 10*(E_start+Estep*i+dE/sqrt(8)-potential(x[j]))/12. - d[2*j]; dnew2[2*j+1] = -d[2*j+1]+10*dE/(12*sqrt(8));
		    
		  dinfnew[2*j] = (E_start+Estep*i-dE/sqrt(8)-potential(x[j]))/12.-dinf[2*j]; dinfnew[2*j+1] = -dinf[2*j+1]-dE/(12*sqrt(8));
		  dsupnew[2*j] = (E_start+Estep*i-dE/sqrt(8)-potential(x[j+1]))/12.-dsup[2*j]; dsupnew[2*j+1] = -dsup[2*j+1]-dE/(12*sqrt(8));	 

		  dinfnew2[2*j] = (E_start+Estep*i+dE/sqrt(8)-potential(x[j]))/12.-dinf[2*j]; dinfnew2[2*j+1] = -dinf[2*j+1]+dE/(12*sqrt(8));
		  dsupnew2[2*j] = (E_start+Estep*i+dE/sqrt(8)-potential(x[j+1]))/12.-dsup[2*j]; dsupnew2[2*j+1] = -dsup[2*j+1]+dE/(12*sqrt(8));

	  }



/*
	  Inv_Tridiagonal_Matrix_complex(dinfnew,dnew,dsupnew,psi,res,num_r+1);
		
	  for(j=0; j<= num_r ; j++) {psi2[2*j] = res[2*j];psi2[2*j+1] = res[2*j+1];}

	  Inv_Tridiagonal_Matrix_complex(dinfnew,dnew2,dsupnew,psi2,res2,num_r+1);
*/




	  Inv_Tridiagonal_Matrix_complex_Numerov(dinfnew,dnew,dsupnew,psi,res,num_r);
		
	  for(j=0; j<= num_r ; j++) {psi2[2*j] = res[2*j];psi2[2*j+1] = res[2*j+1];}

	  Inv_Tridiagonal_Matrix_complex_Numerov(dinfnew2,dnew2,dsupnew2,psi2,res2,num_r);



	  prob = norme(res2,num_r);
      	  prob = prob*dx*pow(dE,4.);

	  fprintf(fel,"%e\t%e\n",E_start+Estep*i,prob);
	  
	  printf("\b\b\b\b\b%5d", i); fflush(stdout); 

	}

	printf("\n");



	free(dnew); free(dnew2); free(dinfnew);
	free(res);free(dsupnew);
	free(dinfnew2);free(dsupnew2);
	free(res2);free(psi2); 
	fclose(fel);

}


void dipole_analysis(double num_w,double dw,double *timet,double *dipole, int nc, int num_t)
{

	int i,j;
	double FFT_re,FFT_im,omega;
	FILE *fhhg;
	
	fhhg = fopen("HHG_spectrum.dat", "w" );
	printf("Working on the bin 00000");

	
	omega = 0;
	for (i = 0 ; i < num_w ; i++)
	{
		
		FFT_re = 0; FFT_im = 0;
		for( j = 0 ; j < nc*num_t ; j++)
		{
			FFT_re += dipole[2*j]*cos(omega*timet[j])-dipole[2*j+1]*sin(omega*timet[j]);
			FFT_im += dipole[2*j]*sin(omega*timet[j])+dipole[2*j+1]*cos(omega*timet[j]);
		}
	
	   fprintf(fhhg,"%e\t%e\t%e\n",omega,FFT_re,FFT_im);
	   printf("\b\b\b\b\b%5d", i); fflush(stdout);
		
	   omega += dw;
	}

    printf("\n"); fclose(fhhg);

}



void projection_analysis(double Estep,double E_start,int num_E,int num_r,double dx,double *psi,double *x)
{


	double prob,prob_re,prob_im,k,delta;
	FILE *fel;
	int i,j,jmin,jmax;


	fel = fopen("electron_spectrum.dat", "w" );
	printf("Working on the bin 00000");

	num_E = 1000;
	Estep = 0.001;	jmin = 7000; jmax = 8000; delta = (double) jmax- (double) jmin;
		
	for(i=0; i<=  num_E ; i++)
	{
	 		
	  k = sqrt(2.0*Estep*i); 
	  prob_re = 0.0; prob_im = 0.0;
          for(j = jmin ; j<= jmax ; j++) 
	  {	
		prob_re +=  psi[2*j]*cos(k*x[j])-psi[2*j+1]*sin(k*x[j]);  
		prob_im +=  psi[2*j]*sin(k*x[j])+psi[2*j+1]*cos(k*x[j]);

	  }

	  prob_re *= dx/delta; prob_im *= dx/delta; 	


	  prob = (prob_re*prob_re+prob_im*prob_im)/(2.0*Pi);


	  fprintf(fel,"%e\t%e\n",Estep*i,prob);
	  printf("\b\b\b\b\b%5d", i); fflush(stdout); 

	}

	printf("\n");

	fclose(fel);


}

void projection_analysis_EV(double dE,double Estep,double E_start,int num_E,int num_r,double dx,double *psi,double *dinf,double *d,double *dsup,double *x)
{


	double prob,prob_re,prob_im,k,delta,CV,Eguess,E_previous,E,ps_re,ps_im;
	double *psi_EV;
	FILE *fel;
	int i,j;

	psi_EV = calloc(2*(num_r+1),sizeof(double));


	fel = fopen("electron_spectrum.dat", "w" );
	printf("Working on the bin 00000");

	num_E = 2000;
	Estep = 0.001;	


	CV = 1E-10; // CV criteria  
	E_previous = 0;
		
	for(i=0; i<= num_E ; i++)
	{

		for(j=0;j<=num_r;j++) {psi_EV[2*j] = 1; psi_EV[2*j+1] = 0.;}
		normalise(psi_EV,num_r); // Initialise psi_EV for Einitialise

	 	Eguess = i*Estep; 
		
		E = Einitialise(psi_EV,dinf,d,dsup,x,Eguess,CV,num_r);
		printf("%e\t%e\n",Eguess,E);


		ps_re = 0; ps_im = 0;
		for(j=0;j<=num_r;j++)
		{
			ps_re += (psi[2*j]*psi_EV[2*j]-psi[2*j+1]*psi_EV[2*j+1])*x[j];
			ps_im += (psi[2*j]*psi_EV[2*j+1]+psi[2*j+1]*psi_EV[2*j])*x[j];
			//fprintf(fel,"%e\t%e\t%e\n",x[j],psi_EV[2*j],psi_EV[2*j+1]);
		}

	 prob = ps_re*ps_re + ps_im*ps_im; 
	 // if ( Eguess != E_previous)
	  //{	
	    fprintf(fel,"%e\t%e\n",E,prob); //E_previous = E;
	  //}
	  //printf("\b\b\b\b\b%5d", i); fflush(stdout); 

	}

	printf("\n");

	fclose(fel); free(psi_EV);


}



