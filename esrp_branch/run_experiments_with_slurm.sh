#!/bin/bash

ESR=false

PMDK=false

MPI_Windows=false

FS=false

######################################################################################
if $ESR == true
then
   processes=(256)
   size=420
   esr_mode="inplace"
   dir="output/size_${size}_${esr_mode}_baseline_slurm"
   [ -d ${dir} ] || mkdir ${dir}
   for p in "${processes[@]}"
   do
      sbatch --nodes=8 -w node[026-028,031,033-036] --partition="private" --output=${dir}/running_results_${p}_processes_slurm.dat sbatch.sh "-g poisson7-${size}" ${p} 0 ${dir}/running_results_${p}_processes.txt "split" "none" 0 -1 "pcg" 0 0 "inplace" ""
   done
fi
######################################################################################
if $checkpoint_pmdk_processes == true
then   
   processes=(20)
   size=420
   esr_mode="checkpoint-nvram-homogenous"
   dir="output/size_${size}_${esr_mode}/recover"
   [ -d ${dir} ] || mkdir ${dir}
   for p in "${processes[@]}"
   do
      sbatch --nodes=1 -w nvram001 --partition="nvram" --exclusive --output=${dir}/running_results_${p}_processes_slurm.dat sbatch.sh "-g poisson7-${size}" ${p} 0 ${dir}/running_results_${p}_processes.txt "split" "none"  4 "pcg" 0 0 ${esr_mode} "/mnt/pmem_ext4/yaniv_pmem_pool/pool"
    
   done
fi


####################################################################################
if $MPI_Windows == true
then
   processes=(257)
   size=420
   esr_mode="checkpoint-nvram-RDMA"
   dir="output/size_${size}_${esr_mode}"
   [ -d ${dir} ] || mkdir ${dir}
   for p in "${processes[@]}"
   do  
      sbatch --nodes=9 -w node[026-028,031,033-036],nvram001 --partition "private" --exclusive --output=${dir}/running_results_${p}_processes_slurm.dat sbatch.sh "-g poisson7-${size}" ${p} 0 ${dir}/running_results_${p}_processes.txt "split" "none" 0 -1 "pcg" 0 0 "checkpoint-nvram-RDMA" "/mnt/pmem_ext4/test-mpi-pmem"
#"/mnt/pmem_ext4/test-mpi-pmem" path to NVRAM dir to create the mpi windows and persist the data on them
   done
fi

######################################################################################
if $FS == true
then
   processes=(256)
   size=420
   esr_mode="checkpoint-disk"
   dir="output/size_${size}_${esr_mode}/"
   [ -d ${dir} ] || mkdir ${dir}
   for p in "${processes[@]}"
   do
      sbatch --nodes=8 -w node[026-028,031,033-036] --partition="private" --exclusive --output=${dir}/running_results_${p}_processes_slurm.dat sbatch.sh "-g poisson7-${size}" ${p} 0 ${dir}/running_results_${p}_processes.txt "split" "none" 0 -1 "pcg" 0 0 "checkpoint-disk" "/mnt/sdb_partition/esr_checkpoint"
 #"/mnt/sdb_partition/esr_checkpoint" path to any Storage file system directory to persist the data
 done
fi




