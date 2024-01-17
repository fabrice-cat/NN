GCC=/usr/bin/gcc 
COMPILER=h5cc
OBJ_TDSE=main.o prop.o tridiag.o h5_tools.o
SRC_TDSE=TDSE/main.c TDSE/prop.c TDSE/tridiag.c HDF5_tools/h5_tools.c
SRC_MXWL=Maxwell/main.c Maxwell/tools.c HDF5_tools/h5_tools.c Maxwell/util.c
OBJ_MXWL=main.o tools.o h5_tools.o util.o
TRG=TDSE.out Maxwell.out
FLAGS=-O2 -std=c99

all: $(TRG)

TDSE.out: $(OBJ_TDSE)
	$(COMPILER) $(FLAGS) -o TDSE.out $(OBJ_TDSE)
	rm $(OBJ_TDSE)

$(OBJ_TDSE): $(SRC_TDSE)
	$(COMPILER) $(FLAGS) -c $(SRC_TDSE)

Maxwell.out: $(OBJ_MXWL) TDSE.out
#	$(COMPILER) $(FLAGS) -I/usr/local/include -o Maxwell.out $(OBJ_MXWL)
#	$(COMPILER) $(FLAGS) -I/usr/local/include -L/usr/local/lib -lfftw3 -lm -o Maxwell.out $(OBJ_MXWL)
	$(GCC) $(FLAGS) -L/usr/local/lib -lfftw3 -lhdf5 -lm -o Maxwell.out $(OBJ_MXWL)
	rm $(OBJ_MXWL)

$(OBJ_MXWL): $(SRC_MXWL) TDSE.out
#	$(COMPILER) $(FLAGS) -I/usr/local/include -c $(SRC_MXWL)
	$(GCC) $(FLAGS) -I/usr/local/include -c $(SRC_MXWL)


clean:
	rm $(TRG)