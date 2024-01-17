#include "h5_tools.h"

/**
 * @brief Reads data from an existing HDF5 file.
 * 
 * @details Provided the hdf5 file, this function loads the data from a dataset
 * of a given type to a buffer.
 *
 * 
 * @param file_id hdf5 file.
 * @param dset Dataset name in quotation marks ("...").
 * @param Datatype Type of the data being loaded: H5T_NATIVE_INT32 for int, 
 * H5T_NATIVE_DOUBLE for double etc.
 * @param buf Pointer to the buffer variable of a given type.
 * @return herr_t status
 * 
 */
herr_t read_h5_data(hid_t file_id, const char * dset, hid_t Datatype, void * buf) {
    hid_t dataset_id;
    herr_t status;
    dataset_id = H5Dopen2(file_id, dset, H5P_DEFAULT);
	status = H5Dread(dataset_id, Datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
	status = H5Dclose(dataset_id);
    return status;
}

/**
 * @brief Create a dataset array in an hdf5 file.
 * 
 * @param file_id hdf5 file.
 * @param dset_name Dataset name in quotation marks ("...").
 * @param datatype Type of the data being loaded: H5T_NATIVE_INT32 for int, 
 * H5T_NATIVE_DOUBLE for double etc.
 * @param rank Rank of the array. 1D array has rank 1, 2D rank 2.
 * @param dims Array of dimensions. For a 1D array, define hsize_t dims[1] and 
 * set dims[0] = N, where N is number of points in the array. Analogically for 2D
 * arrays, but define hsize_t dims[2].
 * @return herr_t status
 */
herr_t create_dataset_array(hid_t file_id, const char * dset_name, hid_t datatype,
    int rank, hsize_t *dims) 
{    
    herr_t status;
    hid_t dataspace_id, dataset_id;
    dataspace_id = H5Screate_simple(rank, dims, NULL);
	dataset_id = H5Dcreate(file_id, dset_name, datatype, 
        dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dclose(dataset_id);
	status = H5Sclose(dataspace_id);
    return status;
}

/**
 * @brief Create a dataset scalar in an hdf5 file.
 * 
 * @param file_id hdf5 file.
 * @param dset_name Dataset name in quotation marks ("...").
 * @param datatype Type of the data being loaded: H5T_NATIVE_INT32 for int, 
 * H5T_NATIVE_DOUBLE for double etc.
 * @return herr_t status
 */
herr_t create_dataset_scalar(hid_t file_id, const char * dset_name, hid_t datatype) 
{    
    herr_t status;
    hid_t dataspace_id, dataset_id;
    dataspace_id = H5Screate(H5S_SCALAR);
	dataset_id = H5Dcreate(file_id, dset_name, datatype, 
        dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dclose(dataset_id);
	status = H5Sclose(dataspace_id);
    return status;
}

/**
 * @brief Write data to h5 file.
 * 
 * @details Creates a new dataset and writes new data to it.
 * 
 * @param file_id hdf5 file.
 * @param dset Dataset name in quotation marks ("...").
 * @param Datatype Type of the data being loaded: H5T_NATIVE_INT32 for int, 
 * H5T_NATIVE_DOUBLE for double etc.
 * @param buf Pointer to the buffer variable of a given type.
 * @return herr_t status 
 */
herr_t write_to_h5(hid_t file_id, const char * dset, hid_t Datatype, void * buf) {
    hid_t dataset_id;
    herr_t status;
    dataset_id = H5Dopen2(file_id, dset, H5P_DEFAULT);
	status = H5Dwrite(dataset_id, Datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
	status = H5Dclose(dataset_id);
    return status;
}

/**
 * @brief Create an h5 file.
 * 
 * @param h5_filename Name of the hdf5 file. 
 * @return hid_t status
 */
hid_t create_h5_file(const char * h5_filename) {
    return H5Fcreate(h5_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

/**
 * @brief Create a group in an h5 file. 
 * 
 * @param file_id hdf5 file.
 * @param group_name Name of the group to be created.
 * @return herr_t status
 */
herr_t create_group(hid_t file_id, const char * group_name) {
    hid_t group_id;
    group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    return H5Gclose(group_id);
}

/**
 * @brief Loads dataset from an old hdf5 file and creates identical dataset
 * with the same data inside a new hdf5 file.
 * 
 * @param file_old Old h5 file for copying.
 * @param file_new New h5 file, data is copied there.
 * @param dset Name of the dataset.
 * @param rank Rank of the dataset: 0 - scalar, 1 - 1D array etc.
 * @param dims Dimensions of the array.
 * @param dum_var Dummy variable of type "dtype".
 * @param dtype Type of the data being loaded: H5T_NATIVE_INT32 for int, 
 * H5T_NATIVE_DOUBLE for double etc.
 * @return herr_t status
 */
herr_t load_and_create_dataset(hid_t file_old, hid_t file_new, const char * dset, 
                               int rank, hsize_t * dims, void * dum_var, hid_t dtype) {
    herr_t status;

    status = read_h5_data(file_old, dset, dtype, dum_var);
    if (rank == 0) {
        status = create_dataset_scalar(file_new, dset, dtype);
    } else {
        status = create_dataset_array(file_new, dset, dtype, rank, dims);
    }
    status = write_to_h5(file_new, dset, dtype, dum_var);

    return status;
}

/**
 * @brief Create next h5 file.
 * 
 * @param file_old Old h5 file for copying.
 * @param h5_filename Name of the new h5 file.
 * @param TDSE If true, copies also inputs necessary for the TDSE solver, else
 * function copies data for Maxwell solver only. 
 * @return hid_t status
 */
hid_t create_next_h5_file(hid_t file_old, const char * h5_filename, bool TDSE) {
    hid_t file_new;
    herr_t status;
    hsize_t dims1D[1];

    double dum_dble, *dum_arr;
    int dum_int, N_total;

    file_new = create_h5_file(h5_filename);
    status = create_group(file_new, "inputs");
    status = create_group(file_new, "outputs");

    status = load_and_create_dataset(file_old, file_new, "/inputs/N_total", 0, dims1D, &N_total, H5T_NATIVE_INT32);
    dum_arr = (double*)calloc(N_total, sizeof(double));
    dims1D[0] = N_total;
    status = load_and_create_dataset(file_old, file_new, "/outputs/tgrid", 1, dims1D, dum_arr, H5T_NATIVE_DOUBLE);
    free(dum_arr);
    status = load_and_create_dataset(file_old, file_new, "/inputs/z", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
    status = load_and_create_dataset(file_old, file_new, "/inputs/dz", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
    status = load_and_create_dataset(file_old, file_new, "/inputs/density", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
    status = load_and_create_dataset(file_old, file_new, "/inputs/n0", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
    status = load_and_create_dataset(file_old, file_new, "/inputs/n2", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
    status = load_and_create_dataset(file_old, file_new, "/inputs/omega", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);

    if (TDSE) {
        status = load_and_create_dataset(file_old, file_new, "/inputs/Eguess", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/num_r", 0, dims1D, &dum_int, H5T_NATIVE_INT32);
        status = load_and_create_dataset(file_old, file_new, "/inputs/num_exp", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/dx", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/E0", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/phi", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/ton", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/toff", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/nc", 0, dims1D, &dum_int, H5T_NATIVE_INT32);
        status = load_and_create_dataset(file_old, file_new, "/inputs/num_t", 0, dims1D, &dum_int, H5T_NATIVE_INT32);
        status = load_and_create_dataset(file_old, file_new, "/inputs/dE", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/Estep", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/num_E", 0, dims1D, &dum_int, H5T_NATIVE_INT32);
        status = load_and_create_dataset(file_old, file_new, "/inputs/E_start", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/num_w", 0, dims1D, &dum_int, H5T_NATIVE_INT32);
        status = load_and_create_dataset(file_old, file_new, "/inputs/dw", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/omega_1", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/omega_2", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/ground_state_energy", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/E0_1", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/E0_2", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/number_of_cycles_1", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/number_of_cycles_2", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/CEP_1", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/CEP_2", 0, dims1D, &dum_dble, H5T_NATIVE_DOUBLE);
        status = load_and_create_dataset(file_old, file_new, "/inputs/points_per_cycle_for_integration", 0, dims1D, &dum_int, H5T_NATIVE_INT32);
        status = load_and_create_dataset(file_old, file_new, "/inputs/points_per_cycle_for_evaluation", 0, dims1D, &dum_int, H5T_NATIVE_INT32);        
    }

    return file_new;
}

/**
 * @brief Get the h5 object filename
 * 
 * @param file_id HDF5 file.
 * @return char* file name
 */
char * get_h5_filename(hid_t file_id) {
    ssize_t size = H5Fget_name(file_id, NULL, 0);
    char * filename = (char*)calloc(50, sizeof(char));
    H5Fget_name(file_id, filename, size+1);
    return filename;
}