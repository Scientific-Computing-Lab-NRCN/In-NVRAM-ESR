FILE=repeal_main
OUTFILE=repeal_pcg
objects =repeal_mat_vec.o repeal_spmv.o repeal_commstrategy.o repeal_pc.o repeal_io.o repeal_pcg.o repeal_options.o repeal_reconstruction.o repeal_reconstruction_utils.o repeal_save_state.o repeal_recovery.o repeal_input_generation.o repeal_binary_io.o repeal_gsl_spmatrix_additions.o pmem_vector.o persist_gsl_vector_mpi.o 

CC= mpicc
CFLAGS=-I/opt/sw/pmdk/pmdk_1.9/pmdk/1.9/include -I/opt/ohpc/pub/libs/gnu/gsl/2.4/include/ -std=gnu11 -Wall -Wpedantic -O3
LDFLAGS=-L/opt/sw/pmdk/pmdk_1.9/pmdk/1.9/lib -lpmemobj -lpmem -L/opt/ohpc/pub/mpi-win-pm-ext/mpich-gnu/2016/lib -lmpi-pmem-one-sided -L/opt/ohpc/pub/libs/gnu/gsl/2.4/lib -lgslcblas -lm -lgsl -L/opt/ohpc/pub/libs/gnu/openblas/0.2.20/lib/ -lopenblas -fopenmp


$(OUTFILE): $(FILE).o mmio.o $(objects)
	$(CC) -o $(OUTFILE) $(FILE).o mmio.o $(objects) $(LDFLAGS)

$(FILE).o: mmio.h
	$(CC) -c $(CFLAGS) $(FILE).c

mmio.o: mmio.c mmio.h
	$(CC) -c -std=gnu11 mmio.c

$(objects): %.o: %.c %.h repeal_utils.h
	$(CC) -c $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f $(FILE) $(FILE).o mmio.o $(objects)
