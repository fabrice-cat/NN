#include "tools.h"
//#define STD_OUT

/**
 * @brief Unidirectional Pulse Propagation Equation solver
 * 
 * @details Given the source term (ab-initio/ad-hoc), this function solves
 * the UPPE for a given number of steps using an ODE solver. The intermediate 
 * results are saved in a chosen directory.
 * 
 * @param init_file Name of the initial file. 
 * @param steps Number of steps for propagation. 
 * @param source_term Selection of a source term - ab-initio / ad-hoc.
 * @param w_max Maximum undamped frequency for propagation.
 * @param dir Directory for saving intermediate results.
 */
void solve_UPPE(char * init_file, int steps, int source_term, double w_max, const char * dir) {
    hid_t file_i = H5Fopen(init_file, H5F_ACC_RDWR, H5P_DEFAULT);
    herr_t status;

    /***** MAIN LOOP *****/
    for (int i = 0; i < steps; i++) {
        RK4_step(&file_i, source_term, i, w_max, dir);
    }

    status = TDSE_execution(&file_i);

    status = H5Fclose(file_i);
    if (status < 0) {
        printf("Unable to close the file!");
    }
}

/**
 * @brief One step of the Runge-Kutta 4th order solver
 * 
 * @details Yields result of the next step of the propagation of the pulse in 
 * a medium with sources.
 * 
 * @param file_id Input HDF5 file.
 * @param source_term Selection of a source term - ab-initio / ad-hoc.
 * @param step n-th step of the propagation. 
 * @param w_max Maximum harmonic order of the fundamental laser frequency.
 * @param dir Directory for saving intermediate results.
 */
void RK4_step(hid_t * file_id, int source_term, int step, double w_max, const char * dir) {
    /* Field variables */
    // Time grid, frequency (omega) grid
    double *t, *w;
    // Maxwell parameters
    // Step length
    double dz;
    // Density of the medium
    double density;
    // Speed of light in vacuum in a.u.
    double c = 137.;
    // Vacuum permitivitty in a.u.
    double epsilon = 1/(4*acos(-1.));
    // Refractive index of argon at 800nm
    double n0;

    /* FFT variables */
    // Fourier transform of E: \hat{E}
    fftw_complex *E_hat;
    // <grad V> term
    fftw_complex *grad_V;
    // Plan for FFTW
    fftw_plan plan;
    // Input and output field, third power of field FFT
    fftw_complex *E, *E_out, *E_power_3;
    // k coefficients for RK4
    fftw_complex *k1, *k2, *k3, *k4;
    // Initial field FFT
    fftw_complex *E_0_hat;
    // Polarization P = <x>, Fourier transform and acceleration term \ddot{x}
    fftw_complex *P, *P_hat, *P_acc, *P_acc_hat;

    // HDF5 variables
    // Error status
    herr_t status;
    // Init file
    hid_t file_i;
    // File in the next step
    hid_t file_new;
    unsigned long long dims[2];

    // File names
    char filename[50];

    /* Computational parameters */
    // Number of points for field and time
    int N; 
    // Timestep
    double dt;
    // Frequency filter
    double *filter;
    // Temporal filter
    fftw_complex *t_filter;
    // Laser fundamental frequency
    double omega_0;
    // chi^(3) susceptibility for argon (a.u., density dependent -> must be multiplied by density)
    double *chi_3;
    // Refractive index derivative
    double dn;

    // Open the initial HDF5 file
    file_i = *file_id;

    // Load parameters from an initial H5 file
    status = read_h5_data(file_i, "/inputs/N_total", H5T_NATIVE_INT32, &N);
    status = read_h5_data(file_i, "/inputs/omega_1", H5T_NATIVE_DOUBLE, &omega_0);

    dims[0] = N;
    dims[1] = 2;

    // Allocate memory
    t = calloc(N, sizeof(double));
    filter = calloc(N, sizeof(double));
    chi_3 = calloc(N, sizeof(double));
    t_filter = fftw_alloc_complex(N);
    E_hat = fftw_alloc_complex(N);
    E_0_hat = fftw_alloc_complex(N);
    E = fftw_alloc_complex(N);
    E_power_3 = fftw_alloc_complex(N);
    E_out = fftw_alloc_complex(N);
    grad_V = fftw_alloc_complex(N);
    P = fftw_alloc_complex(N);
    P_hat = fftw_alloc_complex(N);
    P_acc = fftw_alloc_complex(N);
    P_acc_hat = fftw_alloc_complex(N);
    k1 = fftw_alloc_complex(N);
    k2 = fftw_alloc_complex(N);
    k3 = fftw_alloc_complex(N);
    k4 = fftw_alloc_complex(N);

    // Load field, time grid, dz and density from an initial H5 file
    status = read_h5_data(file_i, "/outputs/tgrid", H5T_NATIVE_DOUBLE, t);
    status = read_h5_data(file_i, "/outputs/Efield", H5T_NATIVE_DOUBLE, E);
    status = read_h5_data(file_i, "/inputs/dz", H5T_NATIVE_DOUBLE, &dz);
    status = read_h5_data(file_i, "/inputs/density", H5T_NATIVE_DOUBLE, &density);

    // Refractive index at omega_0
    /*if (source_term == 0) {
        n0 = n_acc(omega_0, density, omega_0);
        dn = d_n_acc(omega_0, density, omega_0);
    }
    else { */
    n0 = n(omega_0, density, omega_0);
    // Derivative of n at omega_0
    dn = d_n(omega_0, density, omega_0);
    //}

    // Generate grid of angular frequencies omega (w) for FFT
    w = omegas(N, t[1]-t[0]);

    if (source_term == 1) {
        // Chi^3 coefficient in a.u.
        for (int i = 0; i < N; i++) {
            chi_3[i] = density*chi3(w[i], omega_0);
        }
    }

    // Generate spectral filter
    for (int i = 0; i < N; i++) {
        filter[i] = w[i] < 0. ? bandpass(-w[i], omega_0, w_max*omega_0, 0.1*omega_0, 5, 2, omega_0) : 
                                bandpass(w[i], omega_0, w_max*omega_0, 0.1*omega_0, 5, 2, omega_0);
    }

    // Generate temporal filter
    for (int i = 0; i < N; i++) t_filter[i] = 1. + 0.*I;
    temporal_filter(t_filter, t, N);

    /***** RK4 STEPS *****/

    /*** k1 ***/

    if (source_term == 0) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/grad_pot", H5T_NATIVE_DOUBLE, grad_V);

        // Temporal filter of the acceleration
        dipole_acceleration_filtered(P_acc, grad_V, E, t_filter, N);

        // Perform FFT of the acceleration
        plan = fftw_plan_dft_1d(N, P_acc, P_acc_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the acceleration
        phase_shift(P_acc_hat, w, N, t[N-1], 1);
    }

    if (source_term == 2) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/dipole", H5T_NATIVE_DOUBLE, P);

        // Temporal filter of the P term
        temporal_filter(P, t, N);
        
        // Perform FFT of the P term
        plan = fftw_plan_dft_1d(N, P, P_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the P term
        phase_shift(P_hat, w, N, t[N-1], 1);
    }
    
    // Perform FFT of the field
    plan = fftw_plan_dft_1d(N, E, E_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Phase shift the fields
    phase_shift(E_hat, w, N, t[N-1], 1);

    // 3rd order term: \hat{E^3}, computing third power
    if (source_term == 1) complex_power_3(E, E_power_3, N);

    // Save initial field FFT
    E_0_hat[0] = E_hat[0];

    // k1 coefficients computation (skipping w = 0 due to division by zero error)
    // UPPE with <-grad V> - E term
    if (source_term == 0) {
        for (int i = 1; i < N; i++) {
            // Save initial field FFT
            E_0_hat[i] = E_hat[i];
            // k1 = f(z_n, E_n)
            k1[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k1[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n + 0.5*dz*k1
            E_hat[i] = E_0_hat[i] + 0.5*dz*k1[i];
        }
    // ad-hoc UPPE
    } else if (source_term == 1) {
        for (int i = 1; i < N; i++) {
            // Save initial field FFT
            E_0_hat[i] = E_hat[i];
            // k1 = f(z_n, E_n)
            k1[i] = I*filter[i]*((n(w[i], density, omega_0) - n0)*w[i]/c*E_hat[i] + 
                        w[i]/(2*c)*chi_3[i]*E_power_3[i]);
            // E_n + 0.5*dz*k1
            E_hat[i] = E_0_hat[i] + 0.5*dz*k1[i];
        }
    // UPPE with P = -<x> term
    } else if (source_term == 2) {
        for (int i = 1; i < N; i++) {
            // Save initial field FFT
            E_0_hat[i] = E_hat[i];
            // k1 = f(z_n, E_n)
            k1[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k1[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n + 0.5*dz*k1
            E_hat[i] = E_0_hat[i] + 0.5*dz*k1[i];
        }
    }
    
    // Phase shift the fields
    phase_shift(E_hat, w, N, t[N-1], -1);

    // Do the inverse FFT
    plan = fftw_plan_dft_1d(N, E_hat, E_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Normalize output field
    normalize_field(E_out, N);

    // Save the field to h5 file.
    make_filename(filename, dir, "data_k2_", step);
    printf("Next file: %s \n", filename);
    file_new = create_next_h5_file(file_i, filename, true);
    status = create_dataset_array(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, 2, dims);
    status = write_to_h5(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, E_out);

    // Close H5 file
    status = H5Fclose(file_i);
    file_i = file_new;


    /*** k2 ***/
    // Execute TDSE for an input field
    if (source_term == 0) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/grad_pot", H5T_NATIVE_DOUBLE, grad_V);

        // Temporal filter of the acceleration
        dipole_acceleration_filtered(P_acc, grad_V, E_out, t_filter, N);

        // Perform FFT of the acceleration
        plan = fftw_plan_dft_1d(N, P_acc, P_acc_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the acceleration
        phase_shift(P_acc_hat, w, N, t[N-1], 1);
    }

    if (source_term == 2) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/dipole", H5T_NATIVE_DOUBLE, P);

        // Temporal filter of the P term
        temporal_filter(P, t, N);
        
        // Perform FFT of the P term
        plan = fftw_plan_dft_1d(N, P, P_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the P term
        phase_shift(P_hat, w, N, t[N-1], 1);
    }

    // 3rd order term: \hat{E^3}, computing third power
    if (source_term == 1) complex_power_3(E_out, E_power_3, N);
    
    // k2 coefficients computation
    // UPPE with <-grad V> - E term
    if (source_term == 0) {
        for (int i = 1; i < N; i++) {
            // k2 = f(z_n + dz/2, E_n + dz*k1/2)
            k2[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k2[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n + 0.5*dz*k2
            E_hat[i] = E_0_hat[i] + 0.5*dz*k2[i];
        }
    // ad-hoc UPPE
    } else if (source_term == 1) {
        for (int i = 1; i < N; i++) {
            // k2 = f(z_n + dz/2, E_n + dz*k1/2)
            k2[i] = I*filter[i]*((n(w[i], density, omega_0) - n0)*w[i]/c*E_hat[i] + 
                        w[i]/(2*c)*chi_3[i]*E_power_3[i]);
            // E_n + 0.5*dz*k2
            E_hat[i] = E_0_hat[i] + 0.5*dz*k2[i];
        }
    // UPPE with P = -<x> term
    } else if (source_term == 2) {
        for (int i = 1; i < N; i++) {
            // k2 = f(z_n + dz/2, E_n + dz*k1/2)
            k2[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k2[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n + 0.5*dz*k2
            E_hat[i] = E_0_hat[i] + 0.5*dz*k2[i];
        }
    }
    
    // Phase shift the fields
    phase_shift(E_hat, w, N, t[N-1], -1);

    // Do the inverse FFT
    plan = fftw_plan_dft_1d(N, E_hat, E_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Normalize output field
    normalize_field(E_out, N);

    // Save the field to h5 file.
    make_filename(filename, dir, "data_k3_", step);
    printf("Next file: %s \n", filename);
    file_new = create_next_h5_file(file_i, filename, true);
    status = create_dataset_array(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, 2, dims);
    status = write_to_h5(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, E_out);

    // Close H5 file
    status = H5Fclose(file_i);
    file_i = file_new;

    /*** k3 ***/
    if (source_term == 0) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/grad_pot", H5T_NATIVE_DOUBLE, grad_V);

        // Temporal filter of the acceleration
        dipole_acceleration_filtered(P_acc, grad_V, E_out, t_filter, N);

        // Perform FFT of the acceleration
        plan = fftw_plan_dft_1d(N, P_acc, P_acc_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the acceleration
        phase_shift(P_acc_hat, w, N, t[N-1], 1);
    }

    if (source_term == 2) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/dipole", H5T_NATIVE_DOUBLE, P);

        // Temporal filter of the P term
        temporal_filter(P, t, N);
        
        // Perform FFT of the P term
        plan = fftw_plan_dft_1d(N, P, P_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the P term
        phase_shift(P_hat, w, N, t[N-1], 1);
    }
    
    // 3rd order term: \hat{E^3}, computing third power
    if (source_term == 1) complex_power_3(E_out, E_power_3, N);

    // k3 coefficients computation
    // UPPE with <-grad V> - E term
    if (source_term == 0) {
        for (int i = 1; i < N; i++) {
            // k3 = f(z_n + dz/2, E_n + dz*k2/2)
            k3[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k3[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n + dz*k3
            E_hat[i] = E_0_hat[i] + dz*k3[i];
        }
    // ad-hoc UPPE
    } else if (source_term == 1) {
        for (int i = 1; i < N; i++) {
            // k3 = f(z_n + dz/2, E_n + dz*k2/2)
            k3[i] = I*filter[i]*((n(w[i], density, omega_0) - n0)*w[i]/c*E_hat[i] + 
                        w[i]/(2*c)*chi_3[i]*E_power_3[i]);
            // E_n + dz*k3
            E_hat[i] = E_0_hat[i] + dz*k3[i];
        }
    // UPPE with P = -<x> term
    } else if (source_term == 2) {
        for (int i = 1; i < N; i++) {
            // k3 = f(z_n + dz/2, E_n + dz*k2/2)
            k3[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k3[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n + dz*k3
            E_hat[i] = E_0_hat[i] + dz*k3[i];
        }
    }
    
    // Phase shift the fields
    phase_shift(E_hat, w, N, t[N-1], -1);

    // Do the inverse FFT
    plan = fftw_plan_dft_1d(N, E_hat, E_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Normalize output field
    normalize_field(E_out, N);

    // Save the field to h5 file.
    make_filename(filename, dir, "data_k4_", step);
    printf("Next file: %s \n", filename);
    file_new = create_next_h5_file(file_i, filename, true);
    status = create_dataset_array(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, 2, dims);
    status = write_to_h5(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, E_out);

    // Close H5 file
    status = H5Fclose(file_i);
    file_i = file_new;

    /*** k4 ***/
    
    // Execute TDSE for an input field
    if (source_term == 0) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/grad_pot", H5T_NATIVE_DOUBLE, grad_V);

        // Temporal filter of the acceleration
        dipole_acceleration_filtered(P_acc, grad_V, E_out, t_filter, N);

        // Perform FFT of the acceleration
        plan = fftw_plan_dft_1d(N, P_acc, P_acc_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the acceleration
        phase_shift(P_acc_hat, w, N, t[N-1], 1);
    }

    if (source_term == 2) {
        // Execute TDSE for an input field
        status = TDSE_execution(&file_i);
        
        // Read computed data from h5 file
        status = read_h5_data(file_i, "/outputs/dipole", H5T_NATIVE_DOUBLE, P);

        // Temporal filter of the P term
        temporal_filter(P, t, N);
        
        // Perform FFT of the P term
        plan = fftw_plan_dft_1d(N, P, P_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Phase shift the P term
        phase_shift(P_hat, w, N, t[N-1], 1);
    }

    // 3rd order term: \hat{E^3}, computing third power
    if (source_term == 1) complex_power_3(E_out, E_power_3, N);

    // k4 coefficients computation and final field
    if (source_term == 0) {
        for (int i = 1; i < N; i++) {
            // k4 = f(z_n + dz, E_n + dz*k3)
            k4[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k4[i] = I*filter[i]*density/(2*c*epsilon) * P_acc_hat[i]/w[i] - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n+1 = E_n + dz*(k1 + 2k2 + 2k3 + k4)/6
            E_hat[i] = E_0_hat[i] + dz*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.;
        }
    } else if (source_term == 1) {
        for (int i = 1; i < N; i++) {
            // k4 = f(z_n + dz, E_n + dz*k3)
            k4[i] = I*filter[i]*((n(w[i], density, omega_0) - n0)*w[i]/c*E_hat[i] + 
                        w[i]/(2*c)*chi_3[i]*E_power_3[i]);
            // E_n+1 = E_n + dz*(k1 + 2k2 + 2k3 + k4)/6
            E_hat[i] = E_0_hat[i] + dz*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.;
        }
    } else if (source_term == 2) {
        for (int i = 1; i < N; i++) {
            // k4 = f(z_n + dz, E_n + dz*k3)
            k4[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1)*E_hat[i];
            //k4[i] = I*filter[i]*density/(2*c*epsilon) * (-P_hat[i]*w[i]) - I*w[i]/c*(n0 - 1 + omega_0*dn)*E_hat[i];
            // E_n+1 = E_n + dz*(k1 + 2k2 + 2k3 + k4)/6
            E_hat[i] = E_0_hat[i] + dz*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.;
        }
    }
    
    // Phase shift the fields
    phase_shift(E_hat, w, N, t[N-1], -1);

    // Do the inverse FFT
    plan = fftw_plan_dft_1d(N, E_hat, E_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Normalize output field
    normalize_field(E_out, N);

    // Save the field to h5 file.
    make_filename(filename, dir, "data_k1_", step+1);
    //make_filename(filename, dir, "data_", step+1);
    printf("Next file: %s \n", filename);
    file_new = create_next_h5_file(file_i, filename, true);
    status = create_dataset_array(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, 2, dims);
    status = write_to_h5(file_new, "/outputs/Efield", H5T_NATIVE_DOUBLE, E_out);

    // Close H5 file
    status = H5Fclose(file_i);
    *file_id = file_new;
    
    // Destroy plan for FFTW    
    fftw_destroy_plan(plan);

    // Free memory
    free(t);
    free(w);
    free(filter);
    fftw_free(E_hat);
    fftw_free(E_0_hat);
    fftw_free(E);
    fftw_free(E_out);
    fftw_free(E_power_3);
    fftw_free(grad_V);
    fftw_free(P_acc_hat);
    fftw_free(P);
    fftw_free(P_hat);
    fftw_free(k1);
    fftw_free(k2);
    fftw_free(k3);
    fftw_free(k4);

}

/**
 * @brief Execution of the TDSE code 
 * 
 * @details Executes the TDSE code on a given HDF5 file.
 * 
 * @param file_id HDF5 file for TDSE execution.
 * @return herr_t status
 */
herr_t TDSE_execution(hid_t *file_id) {
    herr_t status;
    // Command buffer
    char cmdbuf[80];
    // File name
    char *f = get_h5_filename(*file_id);
    printf("%s \n", f);
    // Close access to the h5 file so the TDSE.out process can use it
    status = H5Fclose(*file_id);

    /*** Execute task - run TDSE ***/
    snprintf(cmdbuf, sizeof(cmdbuf), "./TDSE.out %s", f);
    system(cmdbuf);

    // Open the h5 file again
    *file_id = H5Fopen(f, H5F_ACC_RDWR, H5P_DEFAULT);
    free(f);
    return status;
}

/**
 * @brief Yields an array of angular frequencies
 * 
 * @details Frequencies in FFT are computed using sample size ```dt``` and 
 * number of steps ```N``` as follows:
 * i = 0..N//2, f[i] = i/(N*dt), f[N//2] = 1/(2*dt) <- (Nyquist frequency)
 * i = N//2..N -> negative frequencies from the lowest to the highest neg. freq.
 * 
 * Note: angular frequencies are obtained by a factor 2pi multiplication.
 * 
 * @param N Number of samples.
 * @param dt Sample size
 * @return double* angular frequencies array
 */
double * omegas(int N, double dt) {
    double *w = (double*)calloc(N, sizeof(double));
    double pi = acos(-1);
    int j = N/2;
    for (int i = 0; i < N/2+1; i++) {
        w[i] = (2*pi)*i/N/dt;
    }
    for (int i = N/2+1; i < N; i++) {
        w[i] = -(2*pi)*j/N/dt;
        j--;
    }

    return w;
}

double sigmoid(double x, double x0, int p, double omega_0) {
    return 1 - 1/(1+exp(-p*(x-x0-omega_0)/omega_0));
}

double super_gaussian(double x, double x0, double sigma, int n) {
    return exp(-pow((x-x0)*(x-x0)/(2*sigma*sigma), n));
}

double highpass(double x, double xc, double x0, int n, double omega_0) {
    if (x < xc) 
        return 0.;
    else if (x >= xc && x <= x0)
        return super_gaussian(x, x0, 0.2*omega_0, n);
    else
        return 1.;
}

double bandpass(double x, double x_min, double x_max, double xc, int p, int n, double omega_0) {
    if (x <= x_max-omega_0)
        return highpass(x, xc, x_min, n, omega_0);
    else 
        return sigmoid(x, x_max, p, omega_0);
}

/**
 * @brief Returns refractive index given the value of frequency.
 * 
 * @details Refractive index is computed via refractive index polynomial interpolation 
 * of a given order given the precomputed expansion coefficients coeffs[N].
 * 
 * @param omega Frequency [a.u.]
 * @param density Number density of gas [a.u.]
 * @param omega_0 Fundamental laser frequency [a.u.]
 * @return refractive index (double)
 */
double n(double omega, double density, double omega_0) {
    double coeffs[] = {110.43398864, 1.40736115, 1104.10992812, -256.6316395,
            3865.88976841, 7904.93152857, 409144.9626315};
    double chi = 0.;
    // If omega is outside the frequency range H = (-4.5, 4.5), then n(omega) = n_0
    if ((omega > 3.5*omega_0) || (omega < -3.5*omega_0)) {
        for (int i = 0; i < 7; i++) {
            //chi += coeffs[i]*pow(omega_0, i);
            chi += coeffs[i]*pow(3.5*omega_0, i);
        }
        return 1 + 0.5*density*chi;
    }
    // Compute n(omega) using polynomial interpolation
    for (int i = 0; i < 7; i++) {
        chi += coeffs[i]*pow(omega, i);
    }

    return 1 + 0.5*density*chi;
}

/**
 * @brief Returns derivative of the refractive index
 * 
 * @param omega Frequency [a.u.]
 * @param density Number density of gas [a.u.]
 * @param omega_0 Fundamental laser frequency [a.u.]
 * @return refractive index (double)
 */
double d_n(double omega, double density, double omega_0) {
    double coeffs[] = {1.40736115, 1104.10992812, -256.6316395,
            3865.88976841, 7904.93152857, 409144.9626315};
    double chi = 0.;
    // If omega is outside the frequency range H = (-4.5, 4.5), then n(omega) = n_0
    if ((omega > 3.5*omega_0) || (omega < -3.5*omega_0)) {
        for (int i = 1; i < 7; i++) {
            chi += i*coeffs[i-1]*pow(omega_0, i-1);
        }
        return 0.5*density*chi;
    }
    // Compute n(omega) using polynomial interpolation
    for (int i = 1; i < 7; i++) {
        chi += i*coeffs[i-1]*pow(omega, i-1);
    }

    return 0.5*density*chi;
}

/**
 * @brief Returns refractive index given the value of frequency for the acceleration
 * form of the dipole.
 * 
 * @details Refractive index is computed via refractive index polynomial interpolation 
 * of a given order given the precomputed expansion coefficients coeffs[N].
 * 
 * @param omega Frequency [a.u.]
 * @param density Number density of gas [a.u.]
 * @param omega_0 Fundamental laser frequency [a.u.]
 * @return refractive index (double)
 */
double n_acc(double omega, double density, double omega_0) {
    double coeffs[] = {119.82768711, 3.05810948, 595.71592911, -436.67531168,
            26141.52267514, 12157.06568557, 133541.14843378};
    double chi = 0.;
    // If omega is outside the frequency range H = (-4.5, 4.5), then n(omega) = n_0
    if ((omega > 3.5*omega_0) || (omega < -3.5*omega_0)) {
        for (int i = 0; i < 7; i++) {
            chi += coeffs[i]*pow(omega_0, i);
        }
        return 1 + 0.5*density*chi;
    }
    // Compute n(omega) using polynomial interpolation
    for (int i = 0; i < 7; i++) {
        chi += coeffs[i]*pow(omega, i);
    }

    return 1 + 0.5*density*chi;
}

/**
 * @brief Returns derivative of the refractive index
 * 
 * @param omega Frequency [a.u.]
 * @param density Number density of gas [a.u.]
 * @param omega_0 Fundamental laser frequency [a.u.]
 * @return refractive index (double)
 */
double d_n_acc(double omega, double density, double omega_0) {
    double coeffs[] = {1.40736115, 1104.10992812, -256.6316395,
            3865.88976841, 7904.93152857, 409144.9626315};
    double chi = 0.;
    // If omega is outside the frequency range H = (-4.5, 4.5), then n(omega) = n_0
    if ((omega > 3.5*omega_0) || (omega < -3.5*omega_0)) {
        for (int i = 1; i < 7; i++) {
            chi += i*coeffs[i-1]*pow(omega_0, i-1);
        }
        return 0.5*density*chi;
    }
    // Compute n(omega) using polynomial interpolation
    for (int i = 1; i < 7; i++) {
        chi += i*coeffs[i-1]*pow(omega, i-1);
    }

    return 0.5*density*chi;
}

/**
 * @brief Returns the value of chi^3 coefficient.
 * 
 * @details Refractive index is computed via refractive index polynomial interpolation 
 * of a given order given the precomputed expansion coefficients coeffs[N].
 * 
 * @param omega Frequency [a.u.]
 * @param omega_0 Fundamental laser frequency [a.u.]
 * @return refractive index (double)
 */
double chi3(double omega, double omega_0) {
    double coeffs[] = {2057.28417969, 0.02635765, 389913.28613281, -23.01363838,
            -59286807.45141602, 5425.12028313, 4.79731653e+09, 
            -533299.84822798, -5.94705113e+10, 25148198.72181273, 
            -4.10932828e+12, -5.64174173e+08, 1.00458591e+14, 
            4.83653897e+09, -9.95403255e+13};
    double chi = 0.;
    // If omega is outside the frequency range H = (-4.5, 4.5), then n(omega) = n_0
    if ((omega > 3.5*omega_0) || (omega < -3.5*omega_0)) {
        for (int i = 0; i < 15; i++) {
            //chi += coeffs[i]*pow(omega_0, i);
            chi += coeffs[i]*pow(3.5*omega_0, i);
        }
        return chi;
    }
    // Compute n(omega) using polynomial interpolation
    for (int i = 0; i < 15; i++) {
        chi += coeffs[i]*pow(omega, i);
    }

    return chi;
}

/**
 * @brief Computes fourier transform of third power of the field.
 * 
 * @param E Electric field.
 * @param out FFT of the third power of the field.
 * @param size Size of the array.
 */
void complex_power_3(fftw_complex * E, fftw_complex * out, int size) {
    fftw_plan plan;
    for (int i = 0; i < size; i++) {
        E[i] = cpow(E[i], 3);
    }
    plan = fftw_plan_dft_1d(size, E, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

/**
 * @brief Does the temporal filtering of the temporal pulse via 
 * error function erf(x).
 * 
 * @note The parameters values were chosen empirically from the behavior of 
 * the filter in the Fourier domain. It dampens the edges by a factor of 1e-5.
 * 
 * @param in Input FFTW_complex array containing the temporal data.
 * @param time Time array corresponing to the input array.
 * @param size Number of points in the arrays.
 */
void temporal_filter(fftw_complex * in, double * time, int size) {
    double dt = time[1]-time[0];
    double t_max = time[size-1];
    // Shift of the erf argument wrt time
    double t_shift = t_max/4.9;
    // Slope of the erf wrt time
    double t_slope = t_max/14.8;
    // Shift of the erf argument wrt time
    double t_shift2 = t_max/6.8;
    // Slope of the erf wrt time
    double t_slope2 = t_max/20.5;
    // Erf filter to be applied on the temporal data
    double filter;
    // Initial integer from which the filtering starts
    int i;
    int i_0 = round((t_max*0.5)/dt); 
    // Applying erf filter for the first part
    for (i = 0; i < i_0; i++) {
        // Error function filter
        filter = (1. + erf((time[i] - t_shift2)/t_slope2))/2.;
        // Application of the erf function filter
        in[i] = in[i]*filter;
    }
    // Applying erf filter for the second part
    for (i = i_0; i < size; i++) {
        // Error function filter
        filter = (1. - erf((time[i] - t_max + t_shift)/t_slope))/2.;
        // Application of the erf function filter
        in[i] = in[i]*filter;
    }
}

/**
 * @brief Multiply the Fourier transform by a phase given by omegas
 * 
 * @param in_out FFTW input and output array.
 * @param w Array of frequencies.
 * @param N Number of points in arrays.
 * @param T Maximum time of the pulse.
 * @param sign Sign of the phase. Must be either +1 or -1. 
 */
void phase_shift(fftw_complex * in_out, double * w, int N, double T, int sign) {
    for (int i = 0; i < N; i++) {
        double arg = sign * I * w[i] * T / 2;
        in_out[i] *= cexp(arg);
    }
}

/**
 * @brief Normalize the field after the consequent application of FFT and IFFT
 * 
 * @param in_out FFTW input and output array.
 * @param N Number of bins in the FFT.
 */
void normalize_field(fftw_complex * in_out, int N) {
    for (int i = 0; i < N; i++)
        in_out[i] = creal(in_out[i])/N + 0.*I;
}

/**
 * @brief Computes dipole acceleration given grad V term and electric field E. 
 * It also applies a temporal filter for resonance damping and getting rid of
 * FFT artifacts.  
 * 
 * @param P_acc Output filtered dipole acceleration.
 * @param grad_V Gradient of effective potential.
 * @param E Electric field.
 * @param t_filter Temporal filter.
 * @param N Number of points.
 */
void dipole_acceleration_filtered(fftw_complex * P_acc, 
                                  fftw_complex * grad_V,
                                  fftw_complex * E,
                                  fftw_complex * t_filter,
                                  int N) 
{
    for (int i = 0; i < N; i++) {
        P_acc[i] = t_filter[i]*(-grad_V[i] - E[i]);
    }
}