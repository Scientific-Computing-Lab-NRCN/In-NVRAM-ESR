#include "repeal_reconstruction.h"

#include <stdio.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>

#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>

#include <stdlib.h>

#include "repeal_commstrategy.h"
#include "repeal_mat_vec.h"
#include "repeal_options.h"
#include "repeal_pcg.h"
#include "repeal_reconstruction_utils.h"
#include "repeal_recovery.h"
#include "repeal_save_state.h"
#include "repeal_spmv.h"
#include "repeal_utils.h"

void pcg_reconstruct_in_place(MPI_Comm *world_comm,

                              // dynamic solver data
                              struct PCG_solver_handle *pcg_handle,

                              // static data
                              const struct repeal_matrix *A,
                              const struct repeal_matrix *P,
                              const gsl_vector *b,

                              // redundant copies etc.
                              struct PCG_state_copy *pcg_state_copy,

                              // user-defined options, utility structs, ...
                              const struct ESR_options *esr_opts,
                              struct ESR_setup *esr_setup,
                              const struct SPMV_context *spmv_ctx) {
  size_t local_size = A->size1;
  size_t global_size = A->size2;

  MPI_Comm comm = *(world_comm);
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  int i_am_broken = esr_setup->rec_role == ROLE_reconstructing ? 1 : 0;
  int my_broken_index;

  enum Reconstruction_role rec_role = esr_setup->rec_role;

  // allocate additional communicators
  MPI_Comm inter_comm, reconstruction_comm;
  setup_intercomm(comm, comm_size, i_am_broken, esr_opts, &inter_comm);
  setup_reconstruction_comm(comm, i_am_broken, esr_opts, &reconstruction_comm);

  // Get broken index
  if (rec_role == ROLE_reconstructing) {
    MPI_Comm_rank(reconstruction_comm, &my_broken_index);
  } else {
    my_broken_index = -1;
  }

  // information about data distribution (elements_per_surviving_process etc.)
  // --> needed for communication
  struct reconstruction_data_distribution *distribution =
      get_data_distribution_after_failure(comm_size, esr_opts, spmv_ctx);
  size_t failed_size = distribution->failed_size;

  // allocate memory needed for reconstruction
  gsl_vector *x_global = NULL;  // global vector x
  gsl_vector *r_global = NULL;  // global vector r
  gsl_vector *p_local_prev =
      NULL;              // search direction from the previous iteration
  gsl_vector *v = NULL;  // utility vector
  gsl_vector *w = NULL;  // utility vector
  if (rec_role == ROLE_reconstructing) {
    x_global = gsl_vector_calloc(global_size);
    r_global = gsl_vector_calloc(global_size);
    p_local_prev = gsl_vector_calloc(local_size);
    v = gsl_vector_alloc(local_size);
    w = gsl_vector_alloc(local_size);
  }

  // additional pointers to improve readability
  gsl_vector *x_local = pcg_handle->x;
  gsl_vector *r_local = pcg_handle->r;
  gsl_vector *z_local = pcg_handle->z;
  gsl_vector *p_local = pcg_handle->p;

  // gather global vectors x and r
  MPI_Request requests[2];
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken, r_local,
                             r_global, requests, spmv_ctx, distribution);
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken, x_local,
                             x_global, requests + 1, spmv_ctx, distribution);
  gather_global_vectors_end(2, requests);

  // retrieve redundant copies of lost scalars
  double scalars[2];  // send/receive-buffer
  if (rec_role == ROLE_not_reconstructing) {
    // copy data into send buffer
    scalars[0] = *(pcg_handle->beta);
    scalars[1] = *(pcg_handle->rz);
  }
  retrieve_scalars(inter_comm, i_am_broken, 2, scalars);
  if (rec_role == ROLE_reconstructing) {
    // copy data from receive buffer
    *(pcg_handle->beta) = scalars[0];
    *(pcg_handle->rz) = scalars[1];
  }

  // retrieve redundant copies of search directions
  retrieve_redundant_vector_copies(
      comm, comm_rank, comm_size, i_am_broken, my_broken_index, A->size1,
      esr_opts, A->minimal_info_with_resilience, p_local, p_local_prev,
      pcg_state_copy->buffer_copy_1, pcg_state_copy->buffer_copy_2,
      spmv_ctx->displacements);

  // reconstruct the lost vectors
  if (rec_role == ROLE_reconstructing) {
    // more detailed time measurements
    double times[9];

    // t0
    times[0] = MPI_Wtime();

    // set up the structs that are needed for the PCG solver
    struct solver_options *solver_opts_sub = get_solver_options_from_values(
        SOLVER_pcg, esr_opts->innertol, esr_opts->innertol, 500000, 0,
        0);  // solvertype, atol, rtol, innertol, max_iter, verbose,
             // with_residual_replacement)
    struct ESR_options *esr_opts_sub =
        get_esr_options();  // default settings: no resilience
    struct NVESR_options *nvesr_opts_sub =
        get_nvesr_options();  // default settings: no nvram
    struct SPMV_context *spmv_ctx_sub =
        get_spmv_context_from_existing_distribution(
            esr_opts->num_broken_nodes, COMM_minimal, local_size, failed_size,
            distribution->elements_per_broken_process,
            distribution->displacements_per_broken_process);

    // set up the subsystems that need to be solved
    struct repeal_matrix *A_sub = NULL, *P_sub = NULL, *A_sub_prec = NULL,
                         *P_sub_prec = NULL;

    // t1
    times[1] = MPI_Wtime();

    // subsystem with P
    create_inner_system(reconstruction_comm, my_broken_index, local_size,
                        failed_size, P, esr_opts, spmv_ctx, distribution,
                        esr_opts_sub, spmv_ctx_sub, PC_none, &P_sub,
                        &P_sub_prec);  // or PC_None?

    // t2
    times[2] = MPI_Wtime();

    // subsystem with A
    create_inner_system(reconstruction_comm, my_broken_index, local_size,
                        failed_size, A, esr_opts, spmv_ctx, distribution,
                        esr_opts_sub, spmv_ctx_sub, PC_BJ, &A_sub, &A_sub_prec);

    // t3
    times[3] = MPI_Wtime();

    // compute z: z_j = p_j - beta_(j-1) * p_(j-1)
    gsl_blas_dcopy(p_local, z_local);
    gsl_blas_daxpy(-1 * (*(pcg_handle->beta)), p_local_prev, z_local);

    // compute v: v = z_j - P * r_j
    gsl_blas_dcopy(z_local, v);
    gsl_spblas_dgemv(CblasNoTrans, -1, P->M, r_global, 1, v);

    // t4
    times[4] = MPI_Wtime();

    // solve: P*r = v for r
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, P_sub, P_sub_prec, v,
                       r_local, solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);

    // t5
    times[5] = MPI_Wtime();

    // compute w: w = b - r - A*x
    gsl_blas_dcopy(b, w);
    gsl_blas_daxpy(-1, r_local, w);
    gsl_spblas_dgemv(CblasNoTrans, -1, A->M, x_global, 1, w);

    // t6
    times[6] = MPI_Wtime();

    // solve: A*x = w for x
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, A_sub, A_sub_prec, w,
                       x_local, solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);

    // t7
    times[7] = MPI_Wtime();

    // free the memory for the subsystems
    repeal_matrix_free(A_sub);
    repeal_matrix_free(P_sub);
    repeal_matrix_free(A_sub_prec);
    repeal_matrix_free(P_sub_prec);
    free_solver_options(solver_opts_sub);
    free_esr_options(esr_opts_sub);
    free_spmv_context(spmv_ctx_sub);

    // t8
    times[8] = MPI_Wtime();

    // time measurement output
    // using one of the replacement nodes to print
    if (esr_opts->verbose_reconstruction &&
        comm_rank == esr_opts->reconstruction_output_rank) {
      // total time: t8-t0
      double time_total = times[8] - times[0];

      // creating the P subsystem: t2-t1
      double time_P_sub_creation = times[2] - times[1];

      // solving the P subsystem: t5-t4
      double time_P_sub_solution = times[5] - times[4];

      // creating the A subsytem: t3-t2
      double time_A_sub_creation = times[3] - times[2];

      // solving the A subsystem: t7-t6
      double time_A_sub_solution = times[7] - times[6];

      char *time_results =
          "\nTime measurements taken on rank %d:\n"
          "Total time to create and solve the inner systems: %.5f seconds\n"
          "Creating the subsystem for P: %.5f seconds\n"
          "Solving the subsystem for P: %.5f seconds\n"
          "Creating the subsystem for A: %.5f seconds\n"
          "Solving the subsystem for A: %.5f seconds\n\n";
      printf(time_results,
             comm_rank,  // global rank of the output node
             time_total, time_P_sub_creation, time_P_sub_solution,
             time_A_sub_creation, time_A_sub_solution);
    }
  }

  // free memory
  if (rec_role == ROLE_reconstructing) {
    gsl_vector_free(x_global);
    gsl_vector_free(r_global);
    gsl_vector_free(p_local_prev);
    gsl_vector_free(v);
    gsl_vector_free(w);

    MPI_Comm_free(&reconstruction_comm);
  }
  free_data_distribution_after_failure(distribution);
  MPI_Comm_free(&inter_comm);  // also works as a barrier for time measurement
}

void pcg_reconstruct_from_memory_checkpoint(
    MPI_Comm *world_comm,

    // dynamic solver data
    struct PCG_solver_handle *pcg_handle,

    // redundant copies etc.
    struct PCG_state_copy *pcg_state_copy,

    // user-defined options, utility structs, ...
    const struct ESR_options *esr_opts, struct ESR_setup *esr_setup,
    const struct SPMV_context *spmv_ctx) {
  MPI_Comm comm = *(world_comm);
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  if (esr_setup->rec_role == ROLE_reconstructing) {
    // I am a reconstruction node, need to receive my checkpointed data from a
    // buddy

    // find out which rank should send me the checkpointed data
    int buddy = -1;
    for (int i = esr_opts->redundant_copies - 1; i >= 0; --i) {
      if (!(is_broken(esr_opts->buddies_that_save_me[i], esr_opts))) {
        buddy = esr_opts->buddies_that_save_me[i];
        break;
      }
    }
    if (buddy == -1)
      failure(
          "Did not find a surviving buddy to retrieve the checkpointed data "
          "from");

    MPI_Request requests[5];

    // receive scalars
    double scalars[2];
    MPI_Irecv(scalars, 2, MPI_DOUBLE, buddy, CHECKPOINT_scalars, comm,
              requests);

    // receive vectors
    int local_size = spmv_ctx->elements_per_process[comm_rank];
    MPI_Irecv(pcg_handle->x->data, local_size, MPI_DOUBLE, buddy, CHECKPOINT_x,
              comm, requests + 1);
    MPI_Irecv(pcg_handle->r->data, local_size, MPI_DOUBLE, buddy, CHECKPOINT_r,
              comm, requests + 2);
    MPI_Irecv(pcg_handle->p->data, local_size, MPI_DOUBLE, buddy, CHECKPOINT_p,
              comm, requests + 3);
    MPI_Irecv(pcg_handle->z->data, local_size, MPI_DOUBLE, buddy, CHECKPOINT_z,
              comm, requests + 4);

    MPI_Waitall(5, requests, MPI_STATUSES_IGNORE);

    // copy scalars from receive buffer
    *(pcg_handle->beta) = scalars[0];
    *(pcg_handle->rz) = scalars[1];
  }

  else {
    // I am a surviving node
    // check if I need to send data to some reconstruction node

    int broken_rank;
    int *buddies = malloc(esr_opts->redundant_copies * sizeof(int));
    for (int broken_index = 0; broken_index < esr_opts->num_broken_nodes;
         ++broken_index) {
      broken_rank = esr_opts->broken_node_ranks[broken_index];
      get_buddies_for_rank(
          broken_rank, comm_size, esr_opts->redundant_copies,
          buddies);  // the ranks that own the checkpoints for broken_rank

      for (int buddy_index = esr_opts->redundant_copies - 1; buddy_index >= 0;
           --buddy_index) {
        if (buddies[buddy_index] == comm_rank) {
          // I have to send data to this broken node

          MPI_Request requests[5];

          // send scalars
          double scalars[2];
          scalars[0] = *(pcg_handle->beta);
          scalars[1] = *(pcg_handle->rz);
          MPI_Isend(scalars, 2, MPI_DOUBLE, broken_rank, CHECKPOINT_scalars,
                    comm, requests);

          // find position and length of vectors
          int buffer_offset =
              esr_opts->checkpoint_recvdispls
                  [broken_rank];  // index in the checkpoint buffer where the
                                  // data belonging to broken_rank starts
          int sendcount = spmv_ctx->elements_per_process[broken_rank];

          // send vectors
          MPI_Isend(pcg_state_copy->checkpoint_x->data + buffer_offset,
                    sendcount, MPI_DOUBLE, broken_rank, CHECKPOINT_x, comm,
                    requests + 1);
          MPI_Isend(pcg_state_copy->checkpoint_r->data + buffer_offset,
                    sendcount, MPI_DOUBLE, broken_rank, CHECKPOINT_r, comm,
                    requests + 2);
          MPI_Isend(pcg_state_copy->checkpoint_p->data + buffer_offset,
                    sendcount, MPI_DOUBLE, broken_rank, CHECKPOINT_p, comm,
                    requests + 3);
          MPI_Isend(pcg_state_copy->checkpoint_z->data + buffer_offset,
                    sendcount, MPI_DOUBLE, broken_rank, CHECKPOINT_z, comm,
                    requests + 4);

          MPI_Waitall(5, requests, MPI_STATUSES_IGNORE);

        } else if (!(is_broken(buddies[buddy_index], esr_opts))) {
          // some other node is taking care of this broken node
          break;
        }
      }
    }

    free(buddies);
  }
}

void pcg_reconstruct_from_disk_checkpoint(
    MPI_Comm *world_comm,
    char * checkpoint_file_name,
    // dynamic solver data
    struct PCG_solver_handle *pcg_handle,

    // user-defined options, utility structs, ...
    struct ESR_setup *esr_setup, const struct SPMV_context *spmv_ctx) {
  MPI_Comm comm = *(world_comm);
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  int global_size = spmv_ctx->buffer->size;

  if (esr_setup->rec_role == ROLE_reconstructing) {
    // open file
    MPI_File fh;
    MPI_File_open(MPI_COMM_SELF, checkpoint_file_name, MPI_MODE_RDONLY,
                  MPI_INFO_NULL, &fh);  // mode: read only

    // set the view so that offsets in the file will be computed as multiples of
    // MPI_DOUBLE
    MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

    // read my checkpointed vectors from file
    // MPI_File_write_at_all(fh, offset, vec->data,
    // spmv_ctx->elements_per_process[comm_rank], MPI_DOUBLE,
    // MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, spmv_ctx->displacements[comm_rank],
                     pcg_handle->x->data,
                     spmv_ctx->elements_per_process[comm_rank], MPI_DOUBLE,
                     MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, global_size + spmv_ctx->displacements[comm_rank],
                     pcg_handle->r->data,
                     spmv_ctx->elements_per_process[comm_rank], MPI_DOUBLE,
                     MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, 2 * global_size + spmv_ctx->displacements[comm_rank],
                     pcg_handle->z->data,
                     spmv_ctx->elements_per_process[comm_rank], MPI_DOUBLE,
                     MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, 3 * global_size + spmv_ctx->displacements[comm_rank],
                     pcg_handle->p->data,
                     spmv_ctx->elements_per_process[comm_rank], MPI_DOUBLE,
                     MPI_STATUS_IGNORE);

    // read the scalars from the file
    // MPI_File_write_at(fh, 4*global_size, pcg_handle->beta, 1, MPI_DOUBLE,
    // MPI_STATUS_IGNORE); 	MPI_File_write_at(fh, 4*global_size+1,
    // pcg_handle->rz,
    // 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, 4 * global_size, pcg_handle->beta, 1, MPI_DOUBLE,
                     MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, 4 * global_size + 1, pcg_handle->rz, 1, MPI_DOUBLE,
                     MPI_STATUS_IGNORE);

    // close file
    MPI_File_close(&fh);
  }
}

void pcg_reconstruct_from_nvram_homogenous_checkpoint(
    // dynamic solver data
    struct PCG_solver_handle *pcg_handle,
    // redundant copies etc.
    struct PCG_state_copy *pcg_state_copy,
    struct ESR_setup *esr_setup) {
  if (esr_setup->rec_role == ROLE_reconstructing) {
    pmem_vector_to_gsl_vector(&(D_RW(pcg_state_copy->persistent_copy)->checkpoint_x), pcg_handle->x);
    pmem_vector_to_gsl_vector(&(D_RW(pcg_state_copy->persistent_copy)->checkpoint_r), pcg_handle->r);
    pmem_vector_to_gsl_vector(&(D_RW(pcg_state_copy->persistent_copy)->checkpoint_z), pcg_handle->z);
    pmem_vector_to_gsl_vector(&(D_RW(pcg_state_copy->persistent_copy)->checkpoint_p), pcg_handle->p);
    
    *(pcg_handle->beta) = D_RO(pcg_state_copy->persistent_copy)->beta;
	  *(pcg_handle->rz) = D_RO(pcg_state_copy->persistent_copy)->rz;
  }
}

void pcg_reconstruct_from_nvram_RDMA_checkpoint(MPI_Comm *world_comm, MPI_Win win, bool win_ram_on, size_t local_size,
  // dynamic solver data
  struct PCG_solver_handle *pcg_handle,
  // redundant copies etc.
  struct PCG_state_copy *pcg_state_copy,
  struct ESR_setup *esr_setup) {
    int comm_rank; //yoni debug
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    int num_of_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
    int win_size = 1;
    if (comm_rank == num_of_ranks-1){
	   win_size = (4 * local_size+2) * sizeof(double)*num_of_ranks; 
   }
    printf("DEBUG: win_size=%d, rank=%d\n", win_size, comm_rank);
    if (win_ram_on) {
        MPI_Aint target_disp = (4*local_size+2)*comm_rank; 
        double* values = NULL;
        MPI_Win_fence(0, win);
        if (esr_setup->rec_role == ROLE_reconstructing) {
          values = get_vector_from_win(comm_rank, win, true, target_disp, local_size);
        }
        MPI_Win_fence(0, win);
        if (esr_setup->rec_role == ROLE_reconstructing) {
          for (int i=0; i<local_size; i++){
          gsl_vector_set(pcg_handle->x,i, values[i]);
          }
        }
      
        if (esr_setup->rec_role == ROLE_reconstructing) {
          free(values);
          values = get_vector_from_win(comm_rank, win, true, target_disp + local_size, local_size);     
        }
        MPI_Win_fence(0, win);
        if (esr_setup->rec_role == ROLE_reconstructing) {
          for (int i=0; i<local_size; i++){
            gsl_vector_set(pcg_handle->r,i, values[i]);
          }
        }

        if (esr_setup->rec_role == ROLE_reconstructing) {
          free(values);
          values = get_vector_from_win(comm_rank, win, true, target_disp + 2*local_size, local_size);
        }
        MPI_Win_fence(0, win);
        if (esr_setup->rec_role == ROLE_reconstructing) {
          for (int i=0; i<local_size; i++){
        gsl_vector_set(pcg_handle->z,i, values[i]);
          }
        }
        if (esr_setup->rec_role == ROLE_reconstructing) {
          free(values);
          values = get_vector_from_win(comm_rank, win, true, target_disp + 3*local_size, local_size);
        }
        MPI_Win_fence(0, win);
        printf("DEBUG: 4\n");
        if (esr_setup->rec_role == ROLE_reconstructing) {
            for (int i=0; i<local_size; i++){
              gsl_vector_set(pcg_handle->p,i, values[i]);
            }
        }
        if (esr_setup->rec_role == ROLE_reconstructing) {
          free(values);
          MPI_Get((pcg_handle->beta), 1, MPI_DOUBLE, num_of_ranks-1, target_disp + 4*local_size, 1, MPI_DOUBLE, win); 
          MPI_Get((pcg_handle->rz), 1, MPI_DOUBLE, num_of_ranks-1, target_disp + 4*local_size+1, 1, MPI_DOUBLE, win);   
        }
        MPI_Win_fence(0, win);
    }
    else
    {
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info, "pmem_is_pmem", "true");
      MPI_Info_set(info, "pmem_allocate_in_ram", "false");
      MPI_Info_set(info, "pmem_mode", "checkpoint");
      MPI_Info_set(info, "pmem_name", "CG");
      double * win_data;
      MPI_Win_allocate_pmem(win_size, sizeof(double), info, MPI_COMM_WORLD, &win_data, &win);
      MPI_Info_free(&info);
      MPI_Aint target_disp = (4*local_size+2)*comm_rank; 
      double* values = NULL;
      MPI_Win_fence_pmem(0, win);
      if (esr_setup->rec_role == ROLE_reconstructing) {
        values = get_vector_from_win(comm_rank, win, false, target_disp, local_size);
      }

      MPI_Win_fence_pmem(0, win);
      if (esr_setup->rec_role == ROLE_reconstructing) {
        for (int i=0; i<local_size; i++){
        gsl_vector_set(pcg_handle->x,i, values[i]);
        }
      }
    
      if (esr_setup->rec_role == ROLE_reconstructing) {
        free(values);
        values = get_vector_from_win(comm_rank, win, false, target_disp + local_size, local_size);     
      }

      MPI_Win_fence_pmem(0, win);
      if (esr_setup->rec_role == ROLE_reconstructing) {
        for (int i=0; i<local_size; i++){
          gsl_vector_set(pcg_handle->r,i, values[i]);
        }
      }

      if (esr_setup->rec_role == ROLE_reconstructing) {
        free(values);
        values = get_vector_from_win(comm_rank, win, false, target_disp + 2*local_size, local_size);
      }
      MPI_Win_fence_pmem(0, win);
      if (esr_setup->rec_role == ROLE_reconstructing) {
        for (int i=0; i<local_size; i++){
      gsl_vector_set(pcg_handle->z,i, values[i]);
        }
      }
      if (esr_setup->rec_role == ROLE_reconstructing) {
        free(values);
        values = get_vector_from_win(comm_rank, win, false, target_disp + 3*local_size, local_size);
      }
      MPI_Win_fence_pmem(0, win);
      printf("DEBUG: 4\n");
      if (esr_setup->rec_role == ROLE_reconstructing) {
          for (int i=0; i<local_size; i++){
            gsl_vector_set(pcg_handle->p,i, values[i]);
          }
      }
      if (esr_setup->rec_role == ROLE_reconstructing) {
        free(values);
        MPI_Get_pmem((pcg_handle->beta), 1, MPI_DOUBLE, num_of_ranks-1, target_disp + 4*local_size, 1, MPI_DOUBLE, win); 
        MPI_Get_pmem((pcg_handle->rz), 1, MPI_DOUBLE, num_of_ranks-1, target_disp + 4*local_size+1, 1, MPI_DOUBLE, win);   
      }
    MPI_Win_fence_pmem(0, win);
    }
} 
 
void pcg_reconstruct_inplace_from_nvram_RDMA() {
 // TO DO YONI
}

void pcg_reconstruct_inplace_from_nvram_homogenous(
    MPI_Comm *world_comm,
    PMEMobjpool* pool,

    // dynamic solver data
    struct PCG_solver_handle *pcg_handle,

    // static data
    const struct repeal_matrix *A, const struct repeal_matrix *P,
    const gsl_vector *b,

    // redundant copies etc.
    struct PCG_state_copy *pcg_state_copy,

    // user-defined options, utility structs, ...
    const struct ESR_options *esr_opts, struct ESR_setup *esr_setup,
    const struct SPMV_context *spmv_ctx) {
  size_t local_size = A->size1;
  size_t global_size = A->size2;

  MPI_Comm comm = *(world_comm);
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  int i_am_broken = esr_setup->rec_role == ROLE_reconstructing ? 1 : 0;
  int my_broken_index;

  enum Reconstruction_role rec_role = esr_setup->rec_role;

  // allocate additional communicators
  MPI_Comm inter_comm, reconstruction_comm;
  setup_intercomm(comm, comm_size, i_am_broken, esr_opts, &inter_comm);
  setup_reconstruction_comm(comm, i_am_broken, esr_opts, &reconstruction_comm);

  // Get broken index
  if (rec_role == ROLE_reconstructing) {
    MPI_Comm_rank(reconstruction_comm, &my_broken_index);
  } else {
    my_broken_index = -1;
  }

  // information about data distribution (elements_per_surviving_process etc.)
  // --> needed for communication
  struct reconstruction_data_distribution *distribution =
      get_data_distribution_after_failure(comm_size, esr_opts, spmv_ctx);
  size_t failed_size = distribution->failed_size;

  // allocate memory needed for reconstruction
  gsl_vector *x_global = NULL;  // global vector x
  gsl_vector *r_global = NULL;  // global vector r
  gsl_vector *p_local_prev =
      NULL;              // search direction from the previous iteration
  gsl_vector *v = NULL;  // utility vector
  gsl_vector *w = NULL;  // utility vector
  if (rec_role == ROLE_reconstructing) {
    x_global = gsl_vector_calloc(global_size);
    r_global = gsl_vector_calloc(global_size);
    p_local_prev = gsl_vector_calloc(local_size);
    v = gsl_vector_alloc(local_size);
    w = gsl_vector_alloc(local_size);
  }

  // additional pointers to improve readability
  gsl_vector *x_local = pcg_handle->x;
  gsl_vector *r_local = pcg_handle->r;
  gsl_vector *z_local = pcg_handle->z;
  gsl_vector *p_local = pcg_handle->p;

  // gather global vectors x and r
  MPI_Request requests[2];
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken, r_local,
                             r_global, requests, spmv_ctx, distribution);
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken, x_local,
                             x_global, requests + 1, spmv_ctx, distribution);
  gather_global_vectors_end(2, requests);

  // retrieve redundant copies of lost scalars
  double scalars[2];  // send/receive-buffer
  if (rec_role == ROLE_not_reconstructing) {
    // copy data into send buffer
    scalars[0] = *(pcg_handle->beta);
    scalars[1] = *(pcg_handle->rz);
  }
  retrieve_scalars(inter_comm, i_am_broken, 2, scalars);
  if (rec_role == ROLE_reconstructing) {
    // copy data from receive buffer
    *(pcg_handle->beta) = scalars[0];
    *(pcg_handle->rz) = scalars[1];
  }

  // retrieve redundant copies of search directions
  retrieve_redundant_vector_copies_from_nvram_homogenous(
      pool, i_am_broken, my_broken_index, p_local, p_local_prev,
      &D_RO(pcg_state_copy->persistent_copy)->nvesr_buffer_copy_1, 
      &D_RO(pcg_state_copy->persistent_copy)->nvesr_buffer_copy_2,
      spmv_ctx);

  // reconstruct the lost vectors
  if (rec_role == ROLE_reconstructing) {
    // more detailed time measurements
    double times[9];

    // t0
    times[0] = MPI_Wtime();

    // set up the structs that are needed for the PCG solver
    struct solver_options *solver_opts_sub = get_solver_options_from_values(
        SOLVER_pcg, esr_opts->innertol, esr_opts->innertol, 500000, 0,
        0);  // solvertype, atol, rtol, innertol, max_iter, verbose,
             // with_residual_replacement)
    struct ESR_options *esr_opts_sub =
        get_esr_options();  // default settings: no resilience
    struct NVESR_options *nvesr_opts_sub =
        get_nvesr_options();  // default settings: no nvram
    struct SPMV_context *spmv_ctx_sub =
        get_spmv_context_from_existing_distribution(
            esr_opts->num_broken_nodes, COMM_minimal, local_size, failed_size,
            distribution->elements_per_broken_process,
            distribution->displacements_per_broken_process);

    // set up the subsystems that need to be solved
    struct repeal_matrix *A_sub = NULL, *P_sub = NULL, *A_sub_prec = NULL,
                         *P_sub_prec = NULL;

    // t1
    times[1] = MPI_Wtime();

    // subsystem with P
    create_inner_system(reconstruction_comm, my_broken_index, local_size,
                        failed_size, P, esr_opts, spmv_ctx, distribution,
                        esr_opts_sub, spmv_ctx_sub, PC_none, &P_sub,
                        &P_sub_prec);  // or PC_None?

    // t2
    times[2] = MPI_Wtime();

    // subsystem with A
    create_inner_system(reconstruction_comm, my_broken_index, local_size,
                        failed_size, A, esr_opts, spmv_ctx, distribution,
                        esr_opts_sub, spmv_ctx_sub, PC_BJ, &A_sub, &A_sub_prec);

    // t3
    times[3] = MPI_Wtime();

    // compute z: z_j = p_j - beta_(j-1) * p_(j-1)
    gsl_blas_dcopy(p_local, z_local);
    gsl_blas_daxpy(-1 * (*(pcg_handle->beta)), p_local_prev, z_local);

    // compute v: v = z_j - P * r_j
    gsl_blas_dcopy(z_local, v);
    gsl_spblas_dgemv(CblasNoTrans, -1, P->M, r_global, 1, v);

    // t4
    times[4] = MPI_Wtime();

    // solve: P*r = v for r
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, P_sub, P_sub_prec, v,
                       r_local, solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);

    // t5
    times[5] = MPI_Wtime();

    // compute w: w = b - r - A*x
    gsl_blas_dcopy(b, w);
    gsl_blas_daxpy(-1, r_local, w);
    gsl_spblas_dgemv(CblasNoTrans, -1, A->M, x_global, 1, w);

    // t6
    times[6] = MPI_Wtime();

    // solve: A*x = w for x
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, A_sub, A_sub_prec, w,
                       x_local, solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);

    // t7
    times[7] = MPI_Wtime();

    // free the memory for the subsystems
    repeal_matrix_free(A_sub);
    repeal_matrix_free(P_sub);
    repeal_matrix_free(A_sub_prec);
    repeal_matrix_free(P_sub_prec);
    free_solver_options(solver_opts_sub);
    free_esr_options(esr_opts_sub);
    free_spmv_context(spmv_ctx_sub);

    // t8
    times[8] = MPI_Wtime();

    // time measurement output
    // using one of the replacement nodes to print
    if (esr_opts->verbose_reconstruction &&
        comm_rank == esr_opts->reconstruction_output_rank) {
      // total time: t8-t0
      double time_total = times[8] - times[0];

      // creating the P subsystem: t2-t1
      double time_P_sub_creation = times[2] - times[1];

      // solving the P subsystem: t5-t4
      double time_P_sub_solution = times[5] - times[4];

      // creating the A subsytem: t3-t2
      double time_A_sub_creation = times[3] - times[2];

      // solving the A subsystem: t7-t6
      double time_A_sub_solution = times[7] - times[6];

      char *time_results =
          "\nTime measurements taken on rank %d:\n"
          "Total time to create and solve the inner systems: %.5f seconds\n"
          "Creating the subsystem for P: %.5f seconds\n"
          "Solving the subsystem for P: %.5f seconds\n"
          "Creating the subsystem for A: %.5f seconds\n"
          "Solving the subsystem for A: %.5f seconds\n\n";
      printf(time_results,
             comm_rank,  // global rank of the output node
             time_total, time_P_sub_creation, time_P_sub_solution,
             time_A_sub_creation, time_A_sub_solution);
    }
  }

  // free memory
  if (rec_role == ROLE_reconstructing) {
    gsl_vector_free(x_global);
    gsl_vector_free(r_global);
    gsl_vector_free(p_local_prev);
    gsl_vector_free(v);
    gsl_vector_free(w);

    MPI_Comm_free(&reconstruction_comm);
  }
  free_data_distribution_after_failure(distribution);
  MPI_Comm_free(&inter_comm);  // also works as a barrier for time measurement
}

void pcg_reconstruct(MPI_Comm *world_comm,
                     PMEMobjpool* pool,
                     MPI_Win win,
                     bool win_ram_on,
                     char * checkpoint_file_path,
                     // dynamic solver data
                     struct PCG_solver_handle *pcg_handle,

                     // static data
                     const struct repeal_matrix *A,
                     const struct repeal_matrix *P, const gsl_vector *b,

                     // redundant copies etc.
                     struct PCG_state_copy *pcg_state_copy,

                     // user-defined options, utility structs, ...
                     const struct ESR_options *esr_opts,
                     struct ESR_setup *esr_setup,
                     const struct SPMV_context *spmv_ctx) {

  int comm_rank;
  printf("DEBUG: pcg reconstruct START\n");
  MPI_Comm_rank(*(world_comm), &comm_rank);
  int local_size = spmv_ctx->elements_per_process[comm_rank];
  if (esr_opts->period && esr_setup->local_data_fate == FATE_keep) {
    pcg_solver_reset_to_saved_state(
        pcg_handle, pcg_state_copy);  // TODO take care of iteration count
  }
  switch (esr_opts->reconstruction_strategy) {
    case REC_inplace:
      pcg_reconstruct_in_place(world_comm, pcg_handle, A, P, b, pcg_state_copy,
                               esr_opts, esr_setup, spmv_ctx);
      break;
    case REC_checkpoint_in_memory:
      pcg_reconstruct_from_memory_checkpoint(world_comm, pcg_handle,
                                             pcg_state_copy, esr_opts,
                                             esr_setup, spmv_ctx);
      break;
    case REC_checkpoint_on_disk:
      pcg_reconstruct_from_disk_checkpoint(world_comm, checkpoint_file_path, pcg_handle, esr_setup,
                                           spmv_ctx);
      break;
    case REC_checkpoint_on_nvram_homogenous:
      pcg_reconstruct_from_nvram_homogenous_checkpoint(
          pcg_handle, pcg_state_copy, esr_setup);
      break;
    case REC_nvesr_homogenous:
      pcg_reconstruct_inplace_from_nvram_homogenous(
          world_comm, pool, pcg_handle, A, P, b, pcg_state_copy, esr_opts,
          esr_setup, spmv_ctx);
    case REC_checkpoint_on_nvram_RDMA:
      pcg_reconstruct_from_nvram_RDMA_checkpoint(
          world_comm, win, win_ram_on, local_size, pcg_handle, pcg_state_copy, esr_setup);
      break;
    case REC_nvesr_RDMA:
      pcg_reconstruct_inplace_from_nvram_RDMA(
          world_comm, pool, pcg_handle, A, P, b, pcg_state_copy, esr_opts,
          esr_setup, spmv_ctx);
    case REC_on_survivors:
      break;

    default:
      failure("Unknown reconstruction strategy");
  }
}

// *************************************************
// PIPELINED Preconditioned Conjugate Gradient
// *************************************************

void pipelined_reconstruct_in_place(
    MPI_Comm *world_comm,

    // dynamic solver data
    struct Pipelined_solver_handle *pipelined_handle,

    // static data
    const struct repeal_matrix *A, const struct repeal_matrix *P,
    const gsl_vector *b,

    // redundant copies etc.
    struct Pipelined_state_copy *pipelined_state_copy,

    // user-defined options, utility structs, ...
    const struct ESR_options *esr_opts, struct ESR_setup *esr_setup,
    const struct SPMV_context *spmv_ctx) {
  // *** reconstruction setup ***

  size_t local_size = A->size1;
  size_t global_size = A->size2;
  const double innertol = esr_opts->innertol;

  MPI_Comm comm = *(world_comm);
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  int i_am_broken = esr_setup->rec_role == ROLE_reconstructing ? 1 : 0;
  int my_broken_index;

  enum Reconstruction_role rec_role = esr_setup->rec_role;

  // allocate additional communicators
  MPI_Comm inter_comm, reconstruction_comm;
  setup_intercomm(comm, comm_size, i_am_broken, esr_opts, &inter_comm);
  setup_reconstruction_comm(comm, i_am_broken, esr_opts, &reconstruction_comm);

  // Get broken index
  if (rec_role == ROLE_reconstructing) {
    MPI_Comm_rank(reconstruction_comm, &my_broken_index);
  } else {
    my_broken_index = -1;
  }

  // information about data distribution (elements_per_surviving_process etc.)
  // --> needed for communication
  struct reconstruction_data_distribution *distribution =
      get_data_distribution_after_failure(comm_size, esr_opts, spmv_ctx);
  size_t failed_size = distribution->failed_size;

  // requests for the asynchronous communication
  MPI_Request requests_a[2];
  MPI_Request requests_b[2];

  // *** allocate memory needed for reconstruction ***

  // buffers for gathering the global vectors
  gsl_vector *buffer_a_1 = NULL;
  gsl_vector *buffer_a_2 = NULL;
  gsl_vector *buffer_b_1 = NULL;
  gsl_vector *buffer_b_2 = NULL;

  // local part of old copy of m
  gsl_vector *m_local_prev = NULL;

  //"modified" vectors for intermediate results
  gsl_vector *m_local_mod = NULL;
  gsl_vector *m_local_prev_mod = NULL;
  gsl_vector *w_local_mod = NULL;
  gsl_vector *w_local_prev_mod = NULL;
  gsl_vector *u_local_mod = NULL;
  gsl_vector *u_local_prev_mod = NULL;
  gsl_vector *b_local_mod = NULL;
  gsl_vector *b_local_prev_mod = NULL;

  // memory only needs to be allocated on replacement nodes
  if (i_am_broken) {
    buffer_a_1 = gsl_vector_alloc(global_size);
    buffer_a_2 = gsl_vector_alloc(global_size);
    buffer_b_1 = gsl_vector_alloc(global_size);
    buffer_b_2 = gsl_vector_alloc(global_size);

    m_local_prev = gsl_vector_alloc(local_size);

    m_local_mod = gsl_vector_alloc(local_size);
    m_local_prev_mod = gsl_vector_alloc(local_size);
    w_local_mod = gsl_vector_alloc(local_size);
    w_local_prev_mod = gsl_vector_alloc(local_size);
    u_local_mod = gsl_vector_alloc(local_size);
    u_local_prev_mod = gsl_vector_alloc(local_size);
    b_local_mod = gsl_vector_alloc(local_size);
    b_local_prev_mod = gsl_vector_alloc(local_size);
  }

  // *** reconstruction ***

  // start gathering w_j and w_(j-1)
  // using additional pointers to "assign names" to the buffers to avoid
  // confusion
  gsl_vector *w_global = buffer_a_1;
  gsl_vector *w_global_prev = buffer_a_2;
  MPI_Request *requests_w = requests_a;
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->w, w_global, requests_w,
                             spmv_ctx, distribution);
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->w_prev, w_global_prev,
                             requests_w + 1, spmv_ctx, distribution);

  // start gathering u_j and u_(j-1)
  gsl_vector *u_global = buffer_b_1;
  gsl_vector *u_global_prev = buffer_b_2;
  MPI_Request *requests_u = requests_b;
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->u, u_global, requests_u,
                             spmv_ctx, distribution);
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->u_prev, u_global_prev,
                             requests_u + 1, spmv_ctx, distribution);

  // create the subsystems
  struct repeal_matrix *A_sub = NULL, *P_sub = NULL, *A_sub_prec = NULL,
                       *P_sub_prec = NULL;
  struct solver_options *solver_opts_sub = NULL;
  struct ESR_options *esr_opts_sub = NULL;
  struct NVESR_options *nvesr_opts_sub = NULL;
  struct SPMV_context *spmv_ctx_sub = NULL;
  if (i_am_broken) {
    // set up the structs that are needed for the PCG solver
    solver_opts_sub = get_solver_options_from_values(
        SOLVER_pcg, innertol, innertol, 500000, 0,
        0);  // solvertype, atol, rtol, max_iter, verbose,
             // with_residual_replacement)
    esr_opts_sub = get_esr_options();  // default settings: no resilience
    nvesr_opts_sub = get_nvesr_options(); // default settings: no nvram
    spmv_ctx_sub = get_spmv_context_from_existing_distribution(
        esr_opts->num_broken_nodes, COMM_minimal, local_size, failed_size,
        distribution->elements_per_broken_process,
        distribution->displacements_per_broken_process);

    // set up the subsystems that need to be solved
    create_inner_system(reconstruction_comm, my_broken_index, local_size,
                        failed_size, P, esr_opts, spmv_ctx, distribution,
                        esr_opts_sub, spmv_ctx_sub, PC_Jacobi,
                        &P_sub, &P_sub_prec);
    create_inner_system(reconstruction_comm, my_broken_index, local_size,
                        failed_size, A, esr_opts, spmv_ctx, distribution,
                        esr_opts_sub, spmv_ctx_sub, PC_BJ,
                        &A_sub, &A_sub_prec);
  }

  // retrieve redundant copies of lost scalars
  double scalars[5];  // send/receive-buffer
  if (!i_am_broken) {
    // copy data into send buffer
    scalars[0] = *(pipelined_handle->alpha);
    scalars[1] = *(pipelined_handle->gamma_prev);
    scalars[2] = *(pipelined_handle->gamma);
    scalars[3] = *(pipelined_handle->delta);
    scalars[4] = *(pipelined_handle->rr);
  }
  retrieve_scalars(inter_comm, i_am_broken, 5, scalars);
  if (i_am_broken) {
    // copy data from receive buffer
    *(pipelined_handle->alpha) = scalars[0];
    *(pipelined_handle->gamma_prev) = scalars[1];
    *(pipelined_handle->gamma) = scalars[2];
    *(pipelined_handle->delta) = scalars[3];
    *(pipelined_handle->rr) = scalars[4];
  }

  // retrieve redundant copies of m_j and m_(j-1)
  retrieve_redundant_vector_copies(
      comm, comm_rank, comm_size, i_am_broken, my_broken_index, A->size1,
      esr_opts, A->minimal_info_with_resilience, pipelined_handle->m,
      m_local_prev, pipelined_state_copy->buffer_copy_1,
      pipelined_state_copy->buffer_copy_2, spmv_ctx->displacements);

  // finish gathering w_j and w_(j-1)
  gather_global_vectors_end(2, requests_w);

  // use gathered vectors w_j and w_(j-1)
  if (i_am_broken) {
    gsl_blas_dcopy(pipelined_handle->m, m_local_mod);
    gsl_blas_dcopy(m_local_prev, m_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, P->M, w_global_prev, 1,
                     m_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, P->M, w_global, 1, m_local_mod);

    // solve: P*w = m for w
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, P_sub, P_sub_prec,
                       m_local_prev_mod, pipelined_handle->w_prev,
                       solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, P_sub, P_sub_prec,
                       m_local_mod, pipelined_handle->w, solver_opts_sub,
                       esr_opts_sub, nvesr_opts_sub, spmv_ctx_sub);
  }

  // global w vectors are no longer needed
  // start gathering r_j and r_(j-1)
  gsl_vector *r_global = w_global;
  gsl_vector *r_global_prev = w_global_prev;
  MPI_Request *requests_r = requests_w;
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->r, r_global, requests_r,
                             spmv_ctx, distribution);
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->r_prev, r_global_prev,
                             requests_r + 1, spmv_ctx, distribution);

  // finish gathering u_j and u_(j-1)
  gather_global_vectors_end(2, requests_u);

  // use gathered vectors u_j and u_(j-1)
  if (i_am_broken) {
    gsl_blas_dcopy(pipelined_handle->w, w_local_mod);
    gsl_blas_dcopy(pipelined_handle->w_prev, w_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, A->M, u_global_prev, 1,
                     w_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, A->M, u_global, 1, w_local_mod);

    // solve: A*u = w for u
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, A_sub, A_sub_prec,
                       w_local_prev_mod, pipelined_handle->u_prev,
                       solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, A_sub, A_sub_prec,
                       w_local_mod, pipelined_handle->u, solver_opts_sub,
                       esr_opts_sub, nvesr_opts_sub, spmv_ctx_sub);
  }

  // global u vectors are no longer needed
  // start gathering x_j and x_(j-1)
  gsl_vector *x_global = u_global;
  gsl_vector *x_global_prev = u_global_prev;
  MPI_Request *requests_x = requests_u;
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->x, x_global, requests_x,
                             spmv_ctx, distribution);
  gather_global_vector_begin(inter_comm, comm_rank, i_am_broken,
                             pipelined_handle->x_prev, x_global_prev,
                             requests_x + 1, spmv_ctx, distribution);

  // finish gathering r_j and r_(j-1)
  gather_global_vectors_end(2, requests_r);

  // use gathered vectors r_j and r_(j-1)
  if (i_am_broken) {
    gsl_blas_dcopy(pipelined_handle->u, u_local_mod);
    gsl_blas_dcopy(pipelined_handle->u_prev, u_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, P->M, r_global_prev, 1,
                     u_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, P->M, r_global, 1, u_local_mod);

    // solve: P*r = u for r
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, P_sub, P_sub_prec,
                       u_local_prev_mod, pipelined_handle->r_prev,
                       solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, P_sub, P_sub_prec,
                       u_local_mod, pipelined_handle->r, solver_opts_sub,
                       esr_opts_sub, nvesr_opts_sub, spmv_ctx_sub);
  }

  // finish gathering x_j and x_(j-1)
  gather_global_vectors_end(2, requests_x);

  // use gathered vectors x_j and x_(j-1)
  if (i_am_broken) {
    gsl_blas_dcopy(b, b_local_mod);
    gsl_blas_dcopy(b, b_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, A->M, x_global_prev, 1,
                     b_local_prev_mod);
    gsl_blas_daxpy(-1, pipelined_handle->r_prev, b_local_prev_mod);
    gsl_spblas_dgemv(CblasNoTrans, -1, A->M, x_global, 1, b_local_mod);
    gsl_blas_daxpy(-1, pipelined_handle->r, b_local_mod);

    // solve: A*x = b for x
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, A_sub, A_sub_prec,
                       b_local_prev_mod, pipelined_handle->x_prev,
                       solver_opts_sub, esr_opts_sub, nvesr_opts_sub,
                       spmv_ctx_sub);
    solve_inner_system(reconstruction_comm, my_broken_index,
                       esr_opts->num_broken_nodes, A_sub, A_sub_prec,
                       b_local_mod, pipelined_handle->x, solver_opts_sub,
                       esr_opts_sub, nvesr_opts_sub, spmv_ctx_sub);
  }

  // on reconstruction nodes, recompute the remaining vectors
  if (i_am_broken) {
    double factor = 1. / *(pipelined_handle->alpha);

    // z_(j-1) = 1/alpha_(j-1) * ( w_(j-1) - w_j )
    gsl_blas_daxpy(factor, pipelined_handle->w_prev,
                   pipelined_handle->z);  // okay because z_local was set to 0
    gsl_blas_daxpy(-factor, pipelined_handle->w, pipelined_handle->z);

    // q_(j-1) = 1/alpha_(j-1) * ( u_(j-1) - u_j )
    gsl_blas_daxpy(factor, pipelined_handle->u_prev, pipelined_handle->q);
    gsl_blas_daxpy(-factor, pipelined_handle->u, pipelined_handle->q);

    // s_(j-1) = 1/alpha_(j-1) * ( r_(j-1) - r_j )
    gsl_blas_daxpy(factor, pipelined_handle->r_prev, pipelined_handle->s);
    gsl_blas_daxpy(-factor, pipelined_handle->r, pipelined_handle->s);

    // p_(j-1) = 1/alpha_(j-1) * ( - x_(j-1) + x_j )
    gsl_blas_daxpy(-factor, pipelined_handle->x_prev, pipelined_handle->p);
    gsl_blas_daxpy(factor, pipelined_handle->x, pipelined_handle->p);
  }

  // *** free memory ***

  if (i_am_broken) {
    gsl_vector_free(buffer_a_1);
    gsl_vector_free(buffer_a_2);
    gsl_vector_free(buffer_b_1);
    gsl_vector_free(buffer_b_2);

    gsl_vector_free(m_local_prev);

    gsl_vector_free(m_local_mod);
    gsl_vector_free(m_local_prev_mod);
    gsl_vector_free(w_local_mod);
    gsl_vector_free(w_local_prev_mod);
    gsl_vector_free(u_local_mod);
    gsl_vector_free(u_local_prev_mod);
    gsl_vector_free(b_local_mod);
    gsl_vector_free(b_local_prev_mod);

    repeal_matrix_free(A_sub);
    repeal_matrix_free(P_sub);
    repeal_matrix_free(A_sub_prec);
    repeal_matrix_free(P_sub_prec);
    free_solver_options(solver_opts_sub);
    free_esr_options(esr_opts_sub);
    free_spmv_context(spmv_ctx_sub);

    MPI_Comm_free(&reconstruction_comm);
  }
  free_data_distribution_after_failure(distribution);
  MPI_Comm_free(&inter_comm);  // collective operation => also serves as
                               // "Barrier" for correct time measurements
}

// Wrapper function (in case we want to add different reconstruction strategies)
void pipelined_reconstruct(MPI_Comm *world_comm,

                           // dynamic solver data
                           struct Pipelined_solver_handle *pipelined_handle,

                           // static data
                           const struct repeal_matrix *A,
                           const struct repeal_matrix *P, const gsl_vector *b,

                           // redundant copies etc.
                           struct Pipelined_state_copy *pipelined_state_copy,

                           // user-defined options, utility structs, ...
                           const struct ESR_options *esr_opts,
                           struct ESR_setup *esr_setup,
                           const struct SPMV_context *spmv_ctx) {
  // for periodic ESR: reset to saved state
  if (esr_opts->period && esr_setup->local_data_fate == FATE_keep) {
    pipelined_solver_reset_to_saved_state(pipelined_handle,
                                          pipelined_state_copy);
  }

  switch (esr_opts->reconstruction_strategy) {
    case REC_inplace:
      pipelined_reconstruct_in_place(world_comm, pipelined_handle, A, P, b,
                                     pipelined_state_copy, esr_opts, esr_setup,
                                     spmv_ctx);
      break;
    default:
      failure("Invalid reconstruction strategy for Pipelined PCG");
  }
}
