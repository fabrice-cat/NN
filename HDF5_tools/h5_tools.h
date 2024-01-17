#ifndef H5_TOOLS_H
#define H5_TOOLS_H

#include "hdf5.h"
#include <stdlib.h>

herr_t read_h5_data(hid_t file_id, const char * dset, hid_t Datatype, void * buf);
herr_t create_dataset_array(hid_t file_id, const char * dset_name, hid_t datatype, 
    int rank, hsize_t *dims);
herr_t create_dataset_scalar(hid_t file_id, const char * dset_name, hid_t datatype);
herr_t write_to_h5(hid_t file_id, const char * dset, hid_t Datatype, void * buf);
hid_t create_h5_file(const char * h5_filename); 
herr_t create_group(hid_t file_id, const char * group_name);
herr_t load_and_create_dataset(hid_t file_old, hid_t file_new, const char * dset, 
                               int rank, hsize_t * dims, void * dum_var, hid_t dtype);
hid_t create_next_h5_file(hid_t file_old, const char * h5_filename, bool TDSE);                               
char * get_h5_filename(hid_t file_id);

#endif