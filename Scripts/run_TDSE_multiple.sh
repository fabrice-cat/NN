DIR=tmp/susceptibilities/Argon_nc=10_H=0_10/

proc=0
Nthreads=4

for h5file in ${DIR}*
do
  let proc=proc+1
  (./TDSE.out $h5file) &
  if (($proc%$Nthreads == 0)); then wait; fi
done

echo "Done."
