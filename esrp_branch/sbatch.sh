#!/bin/bash

#SBATCH --job-name=repeal_pcg
#SBATCH --exclusive

matrixoption=$1
nodes=$2
copies=$3
filename=$4
comm=$5
preconditioner=$6
breaknodes=$7
breakiter=$8
solver=$9
residual_replacement=${10}
period=${11}
esr=${12}
pmem_path=${13}

#mpiexecutable="mpirun"

command="(time ${mpiexecutable} -iface ib0 -n ${nodes} ./repeal_pcg ${matrixoption} -comm ${comm} -solver ${solver} -pc ${preconditioner} -maxit 10 -rtol 1e-8 -blocksize 10 -copies ${copies} -breaknodes ${breaknodes} -breakiter ${breakiter} -with_residual_replacement ${residual_replacement} -period ${period} -esr ${esr} --pmem_path ${13} -verbose) > ${filename} 2>&1"
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi-win-pm-ext/mpich-gnu/2016/lib:YOUR_INSTALLED_MPI_PMEM_EXT_DIR/mpi-win-pm-ext/mpich-gnu/2016/lib:/opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/lib:$LD_LIBRARY_PATH
ml gnu/9.1.0 mpich gsl openblas pmdk1.9

echo $command
eval $command

