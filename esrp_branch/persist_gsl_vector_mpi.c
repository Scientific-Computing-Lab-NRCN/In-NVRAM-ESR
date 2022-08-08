#include "persist_gsl_vector_mpi.h"


//TO DO YONI: MPI_Aint target_disp
MPI_Aint put_gsl_vector_in_win(int comm_rank, int target, MPI_Win_pmem win, bool win_ram_on, long int target_disp, gsl_vector *v, size_t local_size)
{ 
    int num_of_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
    //printf("DEBUG: element index to put is %d\n", target_disp);
    //printf("DEBUG: put vector for rank=%d\n", comm_rank);
    if (win_ram_on) 
    {
       /*
       double val;
       for (int i=0; i< local_size; i++) {       
          val = gsl_vector_get(v,i);
          //if (i==0)
            //printf("DEBUG: put element value is: %f, i=%d, rank=%d, target=%d\n", val, i, comm_rank, target);
          MPI_Put(&val, 1, MPI_DOUBLE, target, target_disp+i, 1, MPI_DOUBLE, win);
        }
        */
        MPI_Put(v->data, local_size, MPI_DOUBLE, target, target_disp, local_size, MPI_DOUBLE, win);
    }
    else
    {  
        /*
        double val;       
        for (int i=0; i< local_size; i++) {     
            val = gsl_vector_get(v,i);
            //if (i==0)
                //printf("DEBUG: put element value is: %f, i=%d, rank=%d\n", val, i, comm_rank);
            MPI_Put_pmem(&val, 1, MPI_DOUBLE, target, target_disp+i, 1, MPI_DOUBLE, win);
            } */
        MPI_Put_pmem(v->data, local_size, MPI_DOUBLE, target, target_disp, local_size, MPI_DOUBLE, win);
    }
    return target_disp+local_size;
}


MPI_Aint put_scalar_in_win(int comm_rank, int target, MPI_Win_pmem win, bool win_ram_on, MPI_Aint target_disp, double value)
{
    int num_of_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
    printf("DEBUG: put scalar value is %f\n", value); 
    if (win_ram_on)
       MPI_Put(&value, 1, MPI_DOUBLE, target, target_disp,1, MPI_DOUBLE, win);
    else
       MPI_Put_pmem(&value, 1, MPI_DOUBLE, target, target_disp,1, MPI_DOUBLE, win);
    return target_disp +1 ;
}

double* get_vector_from_win(int comm_rank, MPI_Win_pmem win, bool win_ram_on, MPI_Aint target_disp, size_t local_size)
{
    int num_of_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
     //printf("DEBUG: get gsl vector from window\n"); 
     //printf("DEBUG: offset is %d\n", target_disp); 
     //gsl_vector *vec = gsl_vector_alloc(local_size);
    double * get_val = (double*)malloc(local_size*sizeof(double));
    printf("DEBUG: element index to get is %d\n", target_disp);
    if (win_ram_on)
       for (int i=0; i< local_size; i++) {    
          MPI_Get(&(get_val[i]), 1, MPI_DOUBLE, num_of_ranks-1, target_disp+i, 1, MPI_DOUBLE, win);
    }
    else
    if (win_ram_on)
       for (int i=0; i< local_size; i++) {    
          MPI_Get_pmem(&(get_val[i]), 1, MPI_DOUBLE, num_of_ranks-1, target_disp+i, 1, MPI_DOUBLE, win);
    }
    return get_val;
}

double get_scalar_from_win(int comm_rank, MPI_Win_pmem win, bool win_ram_on, MPI_Aint target_disp)
{
    int num_of_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
    double get_val;
    if (win_ram_on)
       MPI_Get(&get_val, 1, MPI_DOUBLE, num_of_ranks-1, target_disp, 1, MPI_DOUBLE, win);
    else
       MPI_Get_pmem(&get_val, 1, MPI_DOUBLE, num_of_ranks-1, target_disp, 1, MPI_DOUBLE, win); 
    printf("DEBUG: get scalar value is %f, rank=%d\n", get_val, comm_rank);
    return get_val;
}

