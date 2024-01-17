#!/bin/bash
### SUN grid engine parameters
#$ -N MxwllTDSE
#$ -o Maxwell_TDSE/exit_files/output.txt
#$ -e Maxwell_TDSE/exit_files/error.txt
#$ -pe smp 1 # request no. of cores

### Purge all modules
#module purge
### Load FFTW3 and HDF5
module load fftw
module load hdf5
### Make
#make clean
#make

ITERATIONS=0
SOURCE=0
INIT_F=""
DIR=""
CUTOFF=0

while getopts "f:d:i:s:c:" flag;
do
  case "${flag}" in 
    f) INIT_F=${OPTARG};;
    d) DIR=${OPTARG};;
    i) ITERATIONS=${OPTARG};;
    s) SOURCE=${OPTARG};;
    c) CUTOFF=${OPTARG};;
  esac
done

echo "Initial file: ${INIT_F}"
echo "Directory: ${DIR}"
echo "Number of iterations: ${ITERATIONS}"

if [ $SOURCE = "0" ]; then
    echo "Source term: ab-initio <-grad V> - E"
elif [ $SOURCE = "1" ]; then
    echo "Source term: ad-hoc"
elif [ $SOURCE = "2" ]; then
    echo "Source term: ab-initio <-x>"
else
    echo "Unknown source term. Terminating."
    exit 1
fi

echo "Cutoff: H = ${CUTOFF}"





time ./Maxwell.out ${INIT_F} ${DIR} ${ITERATIONS} ${SOURCE} ${CUTOFF}