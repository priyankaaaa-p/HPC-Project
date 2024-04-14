all: mpi_omp

mpi_omp: main.o mpi_omp_functions.o
	mpicc main.o mpi_omp_functions.o -o mpi_omp -std=c99 -L/usr/local/mpip3/lib -lmpiP -lbfd -liberty -fopenmp -lm

mpi_omp_functions.o: mpi_omp_functions.c
	mpicc -c mpi_omp_functions.c

main.o: main.c
	mpicc -c main.c -std=c99 -fopenmp


clean:
	rm *.o mpi_omp
