#!/bin/bash

TASKS=0
DIR=""

while getopts "d:t:" flag;
do
  case "${flag}" in 
    t) TASKS=${OPTARG};;
    d) DIR=${OPTARG};;
  esac
done


echo "Number of tasks: ${TASKS}"
echo "Directory: ${DIR}"

#$ -cwd
#$ -V
#$ -t 0-$TASKS


./TDSE.out {DIR}data_${SGE_TASK_ID}.h5