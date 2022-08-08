# NVMESR

NVM-ESR (Non-Volatile Memory Exact State Reconstruction)
============

**compiling the code** 
- Compile and install the project of mpi-pmem-ext.
- mpich is required to compile and run NVM-ESR with mpi-pmem-ext (not open_mpi)
- ```module purge'''
- '''module load gnu/9.1.0 gsl/2.4 mpich openblas/0.2.20 pmdk1.9``` 
- make

**Running**

The code supports running in the following algorithms and architectures:
1) checkpointing to disk
2) checkpointing to pmem with PMDK (homogenous architecture of NVRAM)
3) checkpointing to pmem with one-sided mpi (NVRAM forming sub-cluster with RDMA operations)
4) ESR
5) NV-ESR with persisting p to pmem with PMDK (homogenous architecture of NVRAM)
6) NV-ESR with persisting p to pmem with one-sided mpi (NVRAM forming sub-cluster with RDMA operations)

- ```module purge'''
- '''module load gnu/9.1.0 gsl/2.4 mpich openblas/0.2.20 pmdk1.9``` 


