import h5py
import numpy as np

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