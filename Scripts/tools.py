import h5py
import numpy as np
from math import erf

def load_result_data():
    print("Type data .dat file name (without suffix): ")
    dat_file = input() + ".dat"

    print("Type parameters .txt file name (without suffix): ")
    param_file = input() + ".txt"

    print("Type the name of the hdf5 file (without suffix): ")
    h5_filename = input() + ".h5"

    create_hdf5(param_file, dat_file, h5_filename)


def create_hdf5(param_file, dat_file, h5_filename):
    """
    create_hdf5(param_file, dat_file, h5_filename)

    This method loads output data from the 1D-TDSE code and saves it in
    an hdf5 archive compatible with the HHGtoolkit library.

    """

    data = np.genfromtxt(dat_file, delimiter='')
    t = data[:,0]
    E_z = data[:,1]
    E_x = np.zeros(len(E_z))
    A_z = data[:,2]
    A_x = np.zeros(len(A_z))
    population = data[:,3]
    d_z = data[:,4]
    d_x = np.zeros(len(d_z))
    current = data[:,6]

    
    with open(param_file, 'r') as pf:
        contents = pf.readlines()

    ### 1st field parameters
    E0_1 = float(contents[4])
    omega_1 = float(contents[5])
    CEP_1 = float(contents[6])
    eps_1 = 0
    theta_1 = 0
    tau_1 = 0

    ### 2nd field parameters
    E0_2 = 0
    omega_2 = 0
    CEP_2 = 0
    eps_2 = 0
    theta_2 = 0
    tau_2 = 0

    ### Numerical parameters
    dt = t[1]-t[0]
    I_p = 0

    ### Number of cycles of the two pulses
    N_cycl_1 = float(contents[9])
    N_cycl_2 = 0

    ### Number of points for integration per cycle
    N_int = int(contents[10])
    N_pts = int((t[-1]-t[0])/dt)//N_cycl_1

    ### Write the data to hdf5 file
    h5_f = h5py.File(h5_filename, 'a')
    inputs = h5_f.create_group("inputs")
    outputs = h5_f.create_group("outputs")

    outputs.create_dataset('tgrid', data=t)
    outputs.create_dataset('dipole', data=np.transpose(np.array([d_z, d_x])), dtype=np.float64)
    outputs.create_dataset('Afield', data=np.transpose(np.array([A_z, A_x])), dtype=np.float64)
    outputs.create_dataset('Efield', data=np.transpose(np.array([E_z, E_x])), dtype=np.float64)
    outputs.create_dataset('Current', data=current, dtype=np.float64)
    outputs.create_dataset('Population', data=population, dtype=np.float64)

    inputs.create_dataset('omega_1', data=omega_1)
    inputs.create_dataset('omega_2', data=omega_2)
    inputs.create_dataset('ground_state_energy', data=I_p)
    inputs.create_dataset('E0_1', data=E0_1)
    inputs.create_dataset('E0_2', data=E0_2)
    inputs.create_dataset('number_of_cycles_1', data=N_cycl_1)
    inputs.create_dataset('number_of_cycles_2', data=N_cycl_2)
    inputs.create_dataset('CEP_1', data=CEP_1)
    inputs.create_dataset('CEP_2', data=CEP_2)

    inputs.create_dataset('points_per_cycle_for_integration', data=N_int)
    inputs.create_dataset('points_per_cycle_for_evaluation', data=N_pts)

    h5_f.close()

def create_hdf5_from_param(
        h5_filename,
        Eguess = -1.,
        num_r = 10000,
        num_exp = 0,
        dx = 0.2,
        E0 = 0.05,
        omega = 0.0568,
        phi = 0.0,
        ton = 0.5,
        toff = 0.5,
        nc = 10,
        num_t = 500,
        dE = 0.0007,
        Estep = 0.0005,
        num_E = 14000,
        E_start = -0.5,
        num_w = 4500,
        dw = 0.0005699,
        z = 0.,
        dz = 0.,
        density = 0.,
        n0 = 0.,
        n2 = 0.,
        Field = None,
        tgrid = None,
        window_size = 5.
    ):
    """
    
    """
    ### Number of points for integration per cycle
    N_int = num_t
    N_pts = num_t
    
    
    ### 1st field parameters
    if (Field is not None) and (tgrid is not None):
        E0_1 = np.max(np.abs(Field))
        N_total = len(tgrid)
    else:
        N_total = int(window_size*nc*num_t + 1)
        T = 2*np.pi/omega
        dt = T/num_t
        tau_FWHM = nc*T
        tgrid = np.array([dt*i for i in range(N_total)])
        E0_1 = E0
        Field = np.multiply(np.exp(-2*np.log(2)*np.power((tgrid - (window_size/2.)*tau_FWHM)\
                            /(tau_FWHM),2))*E0_1, \
                            np.cos(omega*tgrid+phi*np.pi))
        

    omega_1 = omega
    CEP_1 = phi
    eps_1 = 0
    theta_1 = 0
    tau_1 = 0

    ### 2nd field parameters
    E0_2 = 0
    omega_2 = 0
    CEP_2 = 0
    eps_2 = 0
    theta_2 = 0
    tau_2 = 0

    ### Number of cycles of the two pulses
    N_cycl_1 = nc
    N_cycl_2 = 0


    ### Write the data to hdf5 file
    h5_f = h5py.File(h5_filename, 'a')
    inputs = h5_f.create_group("inputs")
    outputs = h5_f.create_group("outputs")

    outputs.create_dataset('Efield', data=np.transpose(np.array([Field, np.zeros((len(Field)))])), dtype=np.float64)
    outputs.create_dataset('tgrid', data=tgrid, dtype=np.float64)
    
    ### Backward compatibility
    inputs.create_dataset('omega_1', data=omega_1)
    inputs.create_dataset('omega_2', data=omega_2)
    inputs.create_dataset('ground_state_energy', data=0.)
    inputs.create_dataset('E0_1', data=E0_1)
    inputs.create_dataset('E0_2', data=E0_2)
    inputs.create_dataset('number_of_cycles_1', data=N_cycl_1)
    inputs.create_dataset('number_of_cycles_2', data=N_cycl_2)
    inputs.create_dataset('CEP_1', data=CEP_1)
    inputs.create_dataset('CEP_2', data=CEP_2)

    inputs.create_dataset('points_per_cycle_for_integration', data=N_int)
    inputs.create_dataset('points_per_cycle_for_evaluation', data=N_pts)

    ### Create datasets for TDSE inputs
    inputs.create_dataset('Eguess', data=Eguess, dtype=np.float64)
    inputs.create_dataset('num_r', data=num_r, dtype=np.intc)
    inputs.create_dataset('num_exp', data=num_exp, dtype=np.intc)
    inputs.create_dataset('dx', data=dx, dtype=np.float64)
    inputs.create_dataset('E0', data=E0, dtype=np.float64)
    inputs.create_dataset('omega', data=omega, dtype=np.float64)
    inputs.create_dataset('phi', data=phi, dtype=np.float64)
    inputs.create_dataset('ton', data=ton, dtype=np.float64)
    inputs.create_dataset('toff', data=toff, dtype=np.float64)
    inputs.create_dataset('nc', data=nc, dtype=np.intc)
    inputs.create_dataset('num_t', data=num_t, dtype=np.intc)
    inputs.create_dataset('dE', data=dE, dtype=np.float64)
    inputs.create_dataset('Estep', data=Estep, dtype=np.float64)
    inputs.create_dataset('num_E', data=num_E, dtype=np.intc)
    inputs.create_dataset('E_start', data=E_start, dtype=np.float64)
    inputs.create_dataset('num_w', data=num_w, dtype=np.intc)
    inputs.create_dataset('dw', data=dw, dtype=np.float64)
    inputs.create_dataset('N_total', data=N_total, dtype=np.intc)

    ### Create datasets for Maxwell solver
    inputs.create_dataset('z', data = z, dtype = np.float64)
    inputs.create_dataset('dz', data = dz, dtype = np.float64)
    inputs.create_dataset('density', data = density, dtype = np.float64)
    inputs.create_dataset('n0', data = n0, dtype = np.float64)
    inputs.create_dataset('n2', data = n2, dtype = np.float64)


    h5_f.close()

def filter(arr, t, slope_coef=14.8, shift_coef=4.9, slope2_coef = 20.5, shift2_coef = 6.8):
    t_max = t[-1]
    M = len(t)
    shift = t_max/shift_coef
    slope = t_max/slope_coef
    shift2 = t_max/shift2_coef
    slope2 = t_max/slope2_coef
    arr[:M//2] = arr[:M//2] * (1+np.array(list(map(erf, (t[:M//2]-shift2)/(slope2)))))/2
    arr[M//2:] = arr[M//2:] * (1-np.array(list(map(erf, (t[M//2:]-t_max+shift)/(slope)))))/2
    return arr