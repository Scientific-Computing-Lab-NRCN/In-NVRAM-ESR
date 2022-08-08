#include "repeal_reconstruction_utils.h"

#include <assert.h>
#include <stdio.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>
#include <libpmemobj.h>
#include <mpi.h>
#include <stdlib.h>

#include "pmem_vector.h"
#include "repeal_mat_vec.h"
#include "repeal_options.h"
#include "repeal_pc.h"
#include "repeal_pcg.h"

/*
void init_esr_setup(struct ESR_setup * esr_setup)
{
        esr_setup->rec_node_ranks = NULL;
}
*/

struct ESR_setup *create_esr_setup() {
  struct ESR_setup *esr_setup = malloc(sizeof(struct ESR_setup));
  if (!esr_setup) failure("Could not allocate memory for ESR_setup");

  esr_setup->rec_node_ranks = NULL;

  esr_setup->continue_comm = MPI_COMM_NULL;
  esr_setup->reconstruction_comm = MPI_COMM_NULL;

  return esr_setup;
}

void free_esr_setup(struct ESR_setup *esr_setup) {
  if (esr_setup->rec_node_ranks) free(esr_setup->rec_node_ranks);

  if (esr_setup->continue_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&(esr_setup->continue_comm));
    esr_setup->continue_comm = MPI_COMM_NULL;
  }

  if (esr_setup->reconstruction_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&(esr_setup->reconstruction_comm));
    esr_setup->reconstruction_comm = MPI_COMM_NULL;
  }

  free(esr_setup);
}

void configure_reconstruction_inplace(MPI_Comm *world_comm,
                                      const struct ESR_options *esr_options,
                                      struct ESR_setup *esr_setup) {
  int comm_rank, comm_size;
  MPI_Comm_rank(*world_comm, &comm_rank);
  MPI_Comm_size(*world_comm, &comm_size);

  int num_broken_nodes = esr_options->num_broken_nodes;

  esr_setup->node_continues =
      1;  // Node simulating the failure always continues for inplace

  // set the flags
  esr_setup->rec_role = ROLE_not_reconstructing;
  esr_setup->local_data_fate = FATE_keep;
  for (int i = 0; i < num_broken_nodes; i++) {
    if (comm_rank == esr_options->broken_node_ranks[i]) {
      esr_setup->rec_role = ROLE_reconstructing;
      esr_setup->local_data_fate = FATE_erase;
      break;
    }
  }

  // reconstruction nodes are the same as the broken nodes
  esr_setup->rec_node_ranks = (int *)malloc(num_broken_nodes * sizeof(int));
  for (int i = 0; i < num_broken_nodes; i++)
    esr_setup->rec_node_ranks[i] = esr_options->broken_node_ranks[i];
}

void configure_reconstruction(MPI_Comm *world_comm,
                              const struct ESR_options *esr_options,
                              struct ESR_setup *esr_setup) {
  switch (esr_options->reconstruction_strategy) {
    case REC_inplace:
    case REC_checkpoint_in_memory:  // checkpointing needs the same flags as
                                    // in-place reconstruction
    case REC_checkpoint_on_disk:
    case REC_checkpoint_on_nvram_homogenous:
    case REC_nvesr_homogenous:
    case REC_checkpoint_on_nvram_RDMA:
    case REC_nvesr_RDMA:
      configure_reconstruction_inplace(world_comm, esr_options, esr_setup);
      break;
    default:
      failure("unknown reconstruction strategy");
      break;
  }

  /*
  if(esr_options->reconstruction_strategy == REC_inplace)
  {
          configure_reconstruction_inplace(world_comm, esr_options, esr_setup);
  }
  */
}

void retrieve_static_data(MPI_Comm *world_comm,
                          const struct ESR_options *esr_opts,
                          struct ESR_setup *esr_setup) {}

int is_broken(int rank, const struct ESR_options *esr_opts) {
  int num_broken_nodes = esr_opts->num_broken_nodes;
  for (int i = 0; i < num_broken_nodes; ++i)
    if (esr_opts->broken_node_ranks[i] == rank) return 1;
  return 0;
}

// only needed for the pipelined PCG
// should be eliminated at some point
void get_my_deceased_status(int comm_rank, int *i_am_broken,
                            int *my_broken_index,
                            const struct ESR_options *esr_opts) {
  int num_broken_nodes = esr_opts->num_broken_nodes;

  *i_am_broken = 0;
  *my_broken_index = -1;
  for (int i = 0; i < num_broken_nodes; ++i)
    if (esr_opts->broken_node_ranks[i] == comm_rank) {
      *i_am_broken = 1;
      *my_broken_index = i;  // to decide which broken node sends to which
                             // memory location during communication
      break;
    }
}

struct reconstruction_data_distribution *get_data_distribution_after_failure(
    int comm_size, const struct ESR_options *esr_opts,
    const struct SPMV_context *spmv_ctx) {
  int num_broken_nodes = esr_opts->num_broken_nodes;
  int num_surviving_nodes = comm_size - num_broken_nodes;

  // allocate memory
  struct reconstruction_data_distribution *distribution =
      malloc(sizeof(struct reconstruction_data_distribution));
  if (!distribution)
    failure("Could not allocate memory for reconstruction metadata");
  distribution->elements_per_surviving_process =
      malloc(num_surviving_nodes * sizeof(int));
  distribution->displacements_per_surviving_process =
      malloc(num_surviving_nodes * sizeof(int));
  distribution->elements_per_broken_process =
      malloc(num_broken_nodes * sizeof(int));
  distribution->displacements_per_broken_process =
      malloc(num_broken_nodes * sizeof(int));

  // find the data distributions for the surviving nodes only / the replacement
  // nodes only find failed_size (== the total number of vector elements that
  // got lost == the size of the inner linear system)

  distribution->failed_size = 0;

  // replacement nodes + failed_size
  for (int i = 0; i < num_broken_nodes; ++i) {
    distribution->elements_per_broken_process[i] =
        spmv_ctx->elements_per_process[esr_opts->broken_node_ranks[i]];
    distribution->failed_size +=
        spmv_ctx->elements_per_process[esr_opts->broken_node_ranks[i]];
  }
  distribution->displacements_per_broken_process[0] = 0;
  for (int i = 1; i < num_broken_nodes; ++i)
    distribution->displacements_per_broken_process[i] =
        distribution->displacements_per_broken_process[i - 1] +
        distribution->elements_per_broken_process
            [i - 1];  // these should be the displacements inside the subsystem
                      // (NOT the global displacements)

  // surviving nodes
  int index_surviving = 0, index_broken = 0;
  for (int rank = 0; rank < comm_size; ++rank) {
    if (rank == esr_opts->broken_node_ranks
                    [index_broken])  // only need to compare against one broken
                                     // node rank because ranks are ordered
    {
      if (index_broken < num_broken_nodes - 1) index_broken++;
    } else {
      distribution->elements_per_surviving_process[index_surviving] =
          spmv_ctx->elements_per_process[rank];
      distribution->displacements_per_surviving_process[index_surviving] =
          spmv_ctx->displacements[rank];  // these should be the global
                                          // displacements
      index_surviving++;
    }
  }

  // this is needed as dummy input for some MPI calls
  distribution->allgatherv_recvcounts = calloc(num_broken_nodes, sizeof(int));
  distribution->allgatherv_recvdispls = calloc(num_broken_nodes, sizeof(int));

  return distribution;
}

void free_data_distribution_after_failure(
    struct reconstruction_data_distribution *distribution) {
  free(distribution->elements_per_surviving_process);
  free(distribution->displacements_per_surviving_process);
  free(distribution->elements_per_broken_process);
  free(distribution->displacements_per_broken_process);
  free(distribution->allgatherv_recvcounts);
  free(distribution->allgatherv_recvdispls);

  free(distribution);
}

void setup_intercomm(MPI_Comm comm, int comm_size, const int i_am_broken,
                     const struct ESR_options *esr_opts,
                     MPI_Comm *inter_comm_ptr) {
  MPI_Comm local_intra_comm;
  int remote_leader;

  // split comm (=MPI_COMM_WORLD) into two intracommunicators (one for the
  // broken nodes and one for the surviving nodes)
  MPI_Comm_split(comm, i_am_broken, 1, &local_intra_comm);

  // find the rank (based on the original comm) of the leader of the other group
  // (leader is arbitrarily chosen to be the node in the group with the lowest
  // world rank, it actually doesn't matter which node is the leader)
  if (i_am_broken) {
    remote_leader = 0;
    while (is_broken(remote_leader, esr_opts)) remote_leader++;
  } else {
    remote_leader = esr_opts->broken_node_ranks[0];
  }

  // create intercommunicator
  MPI_Intercomm_create(local_intra_comm, 0, comm, remote_leader, 1,
                       inter_comm_ptr);

  MPI_Comm_free(&local_intra_comm);
}

void setup_reconstruction_comm(MPI_Comm comm, const int i_am_broken,
                               const struct ESR_options *esr_opts,
                               MPI_Comm *reconstruction_comm_ptr) {
  if (i_am_broken) {
    int num_broken_nodes = esr_opts->num_broken_nodes;

    // create communicator for the broken nodes only
    MPI_Group world_group, reconstruction_group;
    MPI_Comm_group(comm, &world_group);
    MPI_Group_incl(world_group, num_broken_nodes, esr_opts->broken_node_ranks,
                   &reconstruction_group);
    MPI_Comm_create_group(comm, reconstruction_group, 0,
                          reconstruction_comm_ptr);

    // free memory
    MPI_Group_free(&world_group);
    MPI_Group_free(&reconstruction_group);
  }
}

void gather_global_vector_begin(
    MPI_Comm inter_comm, int comm_rank, const int i_am_broken,
    gsl_vector *vec_local, gsl_vector *vec_global, MPI_Request *request,
    const struct SPMV_context *spmv_ctx,
    const struct reconstruction_data_distribution *distribution) {
  // communicate: gather values for vec from the surviving nodes at each of the
  // broken nodes only initiating the communication here

  if (i_am_broken) {
    // set receive buffer to zero
    gsl_vector_set_zero(vec_global);

    MPI_Iallgatherv(vec_local->data, 0, MPI_DOUBLE, vec_global->data,
                    distribution->elements_per_surviving_process,
                    distribution->displacements_per_surviving_process,
                    MPI_DOUBLE, inter_comm, request);
  } else {
    MPI_Iallgatherv(vec_local->data, spmv_ctx->elements_per_process[comm_rank],
                    MPI_DOUBLE, NULL, distribution->allgatherv_recvcounts,
                    distribution->allgatherv_recvdispls, MPI_DOUBLE, inter_comm,
                    request);
  }
}

void gather_global_vectors_end(int num_requests, MPI_Request *requests) {
  // waiting for some previously initiated communication to finish
  MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
}

void retrieve_scalars(MPI_Comm inter_comm, const int i_am_broken,
                      int num_scalars, double *scalars) {
  // alternative way to implement this:
  // https://www.tutorialspoint.com/cprogramming/c_variable_arguments.htm

  if (i_am_broken) {
    // receive data
    MPI_Bcast(scalars, num_scalars, MPI_DOUBLE, 0, inter_comm);
  } else {
    int inter_comm_rank;
    MPI_Comm_rank(inter_comm, &inter_comm_rank);
    if (inter_comm_rank == 0) {
      // this process actually sends the data
      MPI_Bcast(scalars, num_scalars, MPI_DOUBLE, MPI_ROOT, inter_comm);
    } else {
      // just pretending to participate
      MPI_Bcast(scalars, num_scalars, MPI_DOUBLE, MPI_PROC_NULL, inter_comm);
    }
  }
}

void mark_indices_from_datatype(
    int *num_elements_found, int *local_elements, MPI_Datatype *sendtype,
    const struct SPMV_comm_structure_minimal *comm_info) {
  // decode send datatype to find out which indices that neighbour holds

  /*from the standard: result of MPI_Type_get_contents for MPI_COMBINER_INDEXED:
  count: i[0]
  array_of_blocklengths: i[1] to i[i[0]]
  array_of_displacements: i[i[0]+1] to i[2*i[0]]
  oldtype: d[0]
  and ni = 2*count+1, na = 0, nd = 1
  */

  // decode datatype
  int num_integers, num_addresses, num_datatypes, combiner;
  MPI_Type_get_envelope(*sendtype, &num_integers, &num_addresses,
                        &num_datatypes, &combiner);
  assert(combiner == MPI_COMBINER_INDEXED);
  int *array_of_integers = malloc(num_integers * sizeof(int));
  MPI_Datatype old_datatype;
  MPI_Type_get_contents(*sendtype, num_integers, 0, 1, array_of_integers, NULL,
                        &old_datatype);

  // mark the indices that can be received from this neighbour
  int count = array_of_integers[0];
  int *blocklengths = array_of_integers + 1;
  int *blockdisplacements = array_of_integers + count + 1;

  // this goes over all indices in the datatype without checking if we already
  // reached the limit (correct, but inefficient) - maybe optimize this?
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < blocklengths[i]; ++j) {
      if (local_elements[blockdisplacements[i] + j] == 0) {
        local_elements[blockdisplacements[i] + j] = 1;
        (*num_elements_found)++;
      }
    }
  }

  free(array_of_integers);
}

void retrieve_redundant_vector_copies(
    MPI_Comm comm, int comm_rank, int comm_size, const int i_am_broken,
    const int my_broken_index, const int local_size,
    const struct ESR_options *esr_opts,
    const struct SPMV_comm_structure_minimal *comm_info, gsl_vector *vec_local,
    gsl_vector *vec_local_prev, gsl_vector *copy_1, gsl_vector *copy_2,
    const int *displacements) {
  int num_broken_nodes = esr_opts->num_broken_nodes;

  // used on all nodes
  int *num_communications_required_per_broken_node =
      malloc(num_broken_nodes * sizeof(int));
  int *message_offsets_per_node = malloc(num_broken_nodes * sizeof(int));
  int *ranks_that_should_send_messages = NULL;
  int sum_messages;

  // only meaningful on the broken nodes
  int num_communications_necessary = 0;
  int *neighbours_i_receive_data_from = NULL;

  /***** Broken node: find out who I get my redundant copies from ****/

  if (i_am_broken) {
    /*find out who should send me their redundant copies
    strategy: starting with closest neighbours (next, previous, next but one,
    previous but one, ...), continue until we've found all required indices
    (might potentially cause a lot of redundancy, but considering how we chose
    the buddies it's a reasonable strategy)*/

    int next_neighbour;
    int *local_elements = calloc(local_size, sizeof(int));
    int num_elements_found = 0;

    // find out how many neighbours I need to receive messages from
    for (int i = 0; 1; ++i)  // infinite loop
    {
      next_neighbour =
          (comm_rank + (i / 2 + 1) * ((i % 2) ? -1 : 1) + comm_size) %
          comm_size;  // get rank of next closest neighbour
      if (is_broken(next_neighbour, esr_opts) ||
          comm_info->sendcounts[next_neighbour] ==
              0)  // ignoring nodes that are broken or never got data from me
      {
        continue;
      }

      mark_indices_from_datatype(
          &num_elements_found, local_elements,
          &(comm_info->sendtypes[next_neighbour]),
          comm_info);  // mark and count all indices that can be received from
                       // this neighbour
      num_communications_necessary++;  // count the number of nodes I will
                                       // receive data from

      // potential optimization: if the node cannot send me any elements I
      // didn't already get from other nodes, don't communicate with it

      if (num_elements_found == local_size)  // found all necessary indices
        break;
      if (num_elements_found >
          local_size)  // according to my mental model, that should never happen
        failure("whoops, miscounted somewhere!");
    }  // end infinite loop

    // put information into the appropriate place in the broadcast buffer
    num_communications_required_per_broken_node[my_broken_index] =
        num_communications_necessary;

    // another loop to get the ranks of the neighbours that are supposed to send
    // me data
    neighbours_i_receive_data_from =
        malloc(num_communications_necessary * sizeof(int));
    int index = 0;
    for (int i = 0; index < num_communications_necessary;
         ++i)  // yes, mixing i and index is done on purpose
    {
      next_neighbour =
          (comm_rank + (i / 2 + 1) * ((i % 2) ? -1 : 1) + comm_size) %
          comm_size;
      if (!(is_broken(next_neighbour, esr_opts)) &&
          (comm_info->sendcounts[next_neighbour] != 0))
        neighbours_i_receive_data_from[index++] = next_neighbour;
    }

    free(local_elements);
  }  // end if(i_am_broken)

  /***** broadcast: each broken node tells all the other nodes who should send
   * it data *****/

  MPI_Request *broadcastrequests =
      malloc(num_broken_nodes * sizeof(MPI_Request));

  // broadcast number of required messages
  // could be replaced with alltoallv?
  for (int i = 0; i < num_broken_nodes; ++i) {
    MPI_Ibcast(
        num_communications_required_per_broken_node + i, 1, MPI_INT,
        esr_opts->broken_node_ranks[i], comm,
        broadcastrequests + i);  // start multiple non-blocking broadcasts
  }
  MPI_Waitall(num_broken_nodes, broadcastrequests,
              MPI_STATUSES_IGNORE);  // wait for all the broadcasts to finish

  // prepare for next communication round
  sum_messages = 0;
  for (int i = 0; i < num_broken_nodes; ++i)
    sum_messages += num_communications_required_per_broken_node[i];
  message_offsets_per_node[0] = 0;
  for (int i = 1; i < num_broken_nodes; ++i)
    message_offsets_per_node[i] =
        message_offsets_per_node[i - 1] +
        num_communications_required_per_broken_node[i - 1];
  ranks_that_should_send_messages = malloc(sum_messages * sizeof(int));

  // on broken nodes, put the ranks of the nodes that should send me messages
  // into the broadcast buffer
  if (i_am_broken) {
    int *my_sendbuf = ranks_that_should_send_messages +
                      message_offsets_per_node[my_broken_index];
    for (int i = 0; i < num_communications_necessary; ++i)
      my_sendbuf[i] = neighbours_i_receive_data_from[i];
  }

  // broadcast ranks of nodes that should send the messages
  for (int i = 0; i < num_broken_nodes; ++i)
    MPI_Ibcast(
        ranks_that_should_send_messages + message_offsets_per_node[i],
        num_communications_required_per_broken_node[i], MPI_INT,
        esr_opts->broken_node_ranks[i], comm,
        broadcastrequests + i);  // start multiple non-blocking broadcasts
  MPI_Waitall(num_broken_nodes, broadcastrequests,
              MPI_STATUSES_IGNORE);  // wait for all the broadcasts to finish

  free(broadcastrequests);

  /***** Broken nodes: receive redundant copies *****/

  if (i_am_broken) {
    // allocate memory for communication
    MPI_Request *rcv_requests = malloc(
        2 * num_communications_necessary *
        sizeof(MPI_Request));  // requests for the nonblocking communication
    double *buffer_1 =
        calloc(local_size * num_communications_necessary, sizeof(double));
    double *buffer_2 =
        calloc(local_size * num_communications_necessary, sizeof(double));

    // create appropriate number of (non-blocking) receives
    // using the original sendtypes as receivetypes to reverse the communication
    // from the PCG iterations
    for (int i = 0; i < num_communications_necessary; ++i) {
      MPI_Irecv(buffer_1 + i * local_size, 1,
                comm_info->sendtypes[neighbours_i_receive_data_from[i]],
                neighbours_i_receive_data_from[i], 1, comm,
                &(rcv_requests[2 * i]));  // tag 1 => last search direction
      MPI_Irecv(buffer_2 + i * local_size, 1,
                comm_info->sendtypes[neighbours_i_receive_data_from[i]],
                neighbours_i_receive_data_from[i], 2, comm,
                &(rcv_requests[2 * i +
                               1]));  // tag 2 => last-but-one search direction
    }

    // wait for receives to finish
    MPI_Waitall(2 * num_communications_necessary, rcv_requests,
                MPI_STATUSES_IGNORE);

    // copy values from receive buffers into search directions
    for (int i = 0; i < num_communications_necessary; ++i)
      for (int j = 0; j < local_size; ++j)
        if (buffer_1[i * local_size + j] != 0)
          vec_local->data[j] =
              buffer_1[i * local_size + j];  // elements may get written
                                             // multiple times - optimize?
    for (int i = 0; i < num_communications_necessary; ++i)
      for (int j = 0; j < local_size; ++j)
        if (buffer_2[i * local_size + j] != 0)
          vec_local_prev->data[j] =
              buffer_2[i * local_size + j];  // elements may get written
                                             // multiple times - optimize?

    // free memory
    free(rcv_requests);
    free(buffer_1);
    free(buffer_2);
    free(neighbours_i_receive_data_from);
  }

  /***** Surviving nodes: send redundant copies *****/

  else  // I am not a broken node
  {
    int i_am_sender, begin, end, broken_rank, num_messages_i_have_to_send;

    // count overall number of messages I have to send + allocate requests
    num_messages_i_have_to_send = 0;
    for (int i = 0; i < sum_messages; ++i)
      if (ranks_that_should_send_messages[i] == comm_rank)
        num_messages_i_have_to_send++;
    num_messages_i_have_to_send *=
        2;  // because we need to send 2 messages to each rank
    MPI_Request *send_requests =
        malloc(num_messages_i_have_to_send * sizeof(MPI_Request));

    int requestindex = 0;  // index for the send_requests array

    // for each broken node
    for (int nodeindex = 0; nodeindex < num_broken_nodes; ++nodeindex) {
      // find out if I need to send a message to that node
      i_am_sender = 0;
      begin = message_offsets_per_node[nodeindex];
      end = message_offsets_per_node[nodeindex] +
            num_communications_required_per_broken_node[nodeindex];
      for (int i = begin; i < end; ++i) {
        if (ranks_that_should_send_messages[i] == comm_rank) {
          i_am_sender = 1;
          break;
        }
      }

      // if yes: send messages (using the receive datatype for the given rank to
      // reverse the communication that happens in the PCG iterations)
      if (i_am_sender) {
        broken_rank = esr_opts->broken_node_ranks[nodeindex];
        MPI_Isend(copy_1->data + displacements[broken_rank], 1,
                  comm_info->recvtypes[broken_rank], broken_rank, 1, comm,
                  &(send_requests[requestindex++]));  // tag 1: this is the most
                                                      // recent copy
        MPI_Isend(
            copy_2->data + displacements[broken_rank], 1,
            comm_info->recvtypes[broken_rank], broken_rank, 2, comm,
            &(send_requests[requestindex++]));  // tag 2: this is the older copy
      }
    }  // end for each broken node

    // wait for all sends to finish
    MPI_Waitall(num_messages_i_have_to_send, send_requests,
                MPI_STATUSES_IGNORE);

    free(send_requests);

  }  // end else

  free(num_communications_required_per_broken_node);
  free(ranks_that_should_send_messages);
  free(message_offsets_per_node);
}

void retrieve_redundant_vector_copies_from_nvram_homogenous(
    PMEMobjpool *pool, const int i_am_broken, const int broken_index,
    gsl_vector *vec_local, gsl_vector *vec_local_prev,
    const TOID(struct pmem_vector)* copy_1, const TOID(struct pmem_vector)* copy_2,
    const struct SPMV_context *spmv_ctx) {
  /***** Broken nodes: receive redundant copies *****/
  if (!i_am_broken) {
    // Don't do nothing since we recover all missing information from NVRAM.
    return;
  }
  // I am not a broken node
  pmem_vector_to_gsl_vector(copy_1, vec_local);
  pmem_vector_to_gsl_vector(copy_2, vec_local_prev);
}

void create_inner_system(
    MPI_Comm broken_comm, int my_broken_index, size_t local_size,
    size_t failed_size, const struct repeal_matrix *matrix,
    const struct ESR_options *esr_opts, const struct SPMV_context *spmv_ctx,
    const struct reconstruction_data_distribution
        *distribution,  // structs describing the "current state"
    const struct ESR_options *esr_opts_sub,
    const struct SPMV_context *spmv_ctx_sub,
    const enum PC_type pc_type_sub,  // structs describing the new subsystem
                                     // that should be created
    struct repeal_matrix **matrix_sub,
    struct repeal_matrix *
        *matrix_sub_prec  // output: new system matrix + preconditioner
) {
  int num_broken_nodes = esr_opts->num_broken_nodes;

  // construct submatrix

  gsl_spmatrix *M = matrix->M;  // to improve readability

  // make sure the matrix is stored in compressed column storage
  assert(M->sptype == GSL_SPMATRIX_CCS);

  // count the nonzeros in the submatrix
  int nzmax = 0;
  int broken_rank;
  for (int broken_index = 0; broken_index < num_broken_nodes; ++broken_index) {
    broken_rank = esr_opts->broken_node_ranks[broken_index];
    nzmax += M->p[spmv_ctx->displacements[broken_rank] +
                  spmv_ctx->elements_per_process[broken_rank]] -
             M->p[spmv_ctx->displacements[broken_rank]];  // nonzeros in this
                                                          // part of the matrix
  }

  // allocate submatrix (will be set to all zeros)
  gsl_spmatrix *M_sub = gsl_spmatrix_alloc_nzmax(local_size, failed_size, nzmax,
                                                 GSL_SPMATRIX_CCS);

  // overwrite the matrix data
  int submatrixindex = 0, submatrix_colptrindex = 0;
  int lower, upper;
  int previous_nonzeros = 0;
  for (int broken_index = 0; broken_index < num_broken_nodes; ++broken_index) {
    broken_rank = esr_opts->broken_node_ranks[broken_index];

    // copy the value and row index
    lower = M->p[spmv_ctx->displacements[broken_rank]];
    upper = M->p[spmv_ctx->displacements[broken_rank] +
                 spmv_ctx->elements_per_process[broken_rank]];
    for (int i = lower; i < upper; ++i) {
      // if (my_broken_index == 0) printf("value: %e\n", M->data[i]);
      M_sub->data[submatrixindex] = M->data[i];
      M_sub->i[submatrixindex] = M->i[i];
      // if (my_broken_index == 0) printf("value2: %e\n",
      // M_sub->data[submatrixindex]);
      submatrixindex++;
    }

    // fill in the new column pointers
    lower = spmv_ctx->displacements[broken_rank];
    upper = spmv_ctx->displacements[broken_rank] +
            spmv_ctx->elements_per_process[broken_rank];
    for (int i = lower; i < upper; ++i) {
      M_sub->p[submatrix_colptrindex] = previous_nonzeros;
      previous_nonzeros += M->p[i + 1] - M->p[i];
      submatrix_colptrindex++;
    }
  }
  // fill in the final value in the column pointers
  M_sub->p[submatrix_colptrindex] = previous_nonzeros;

  // set the number of nonzeros to the correct value
  M_sub->nz = previous_nonzeros;

  /*
  //create the local submatrix for solving the inner system
  gsl_spmatrix *M_sub_triplet = gsl_spmatrix_alloc(local_size, failed_size);

  int lower_col_index, upper_col_index, col_index, lower_value_index,
  upper_value_index, value_index, broken_node_rank; for (int broken_node_index =
  0; broken_node_index < num_broken_nodes; ++broken_node_index)
  {
          broken_node_rank = esr_opts->broken_node_ranks[broken_node_index];
  //global rank of the currently examined broken node lower_col_index =
  spmv_ctx->displacements[broken_node_rank]; //begin of column range that
  belongs to this node (inclusive) upper_col_index = lower_col_index +
  spmv_ctx->elements_per_process[broken_node_rank]; //end of column range that
  belongs to this node (exclusive)
          //iterate over all columns in the range
          for (col_index = lower_col_index; col_index < upper_col_index;
  ++col_index)
          {
                  //iterate over all values in that column
                  lower_value_index = M->p[col_index];
                  upper_value_index = M->p[col_index+1];
                  for (value_index = lower_value_index; value_index <
  upper_value_index; ++value_index)
                  {
                          //copy value into new matrix
                          //column indices are adjusted to "compress away" all
  columns that do not belong to broken nodes gsl_spmatrix_set(M_sub_triplet,
  M->i[value_index], col_index - spmv_ctx->displacements[broken_node_rank] +
  distribution->displacements_per_broken_process[broken_node_index],
  M->data[value_index]);
                  }
          }
  }

  //convert to CCS storage
  gsl_spmatrix *M_sub = gsl_spmatrix_ccs(M_sub_triplet);
  gsl_spmatrix_free(M_sub_triplet);
  */

  // create repeal_matrix
  *matrix_sub =
      repeal_matrix_create(broken_comm, my_broken_index, num_broken_nodes,
                           M_sub, COMM_minimal, 1, spmv_ctx_sub, esr_opts_sub);

  // construct preconditioner

  struct PC_options *pc_opts_sub = get_pc_options_from_values(
      pc_type_sub, 10);  // preconditioner type, blocksize
  *matrix_sub_prec =
      get_preconditioner(broken_comm, my_broken_index, num_broken_nodes, M_sub,
                         pc_opts_sub, COMM_minimal, spmv_ctx_sub);
  free_pc_options(pc_opts_sub);
}

void solve_inner_system(
    MPI_Comm broken_comm, const int my_broken_index, const int num_broken_nodes,
    const struct repeal_matrix *M_sub, const struct repeal_matrix *P_sub,
    const gsl_vector *rhs, gsl_vector *vec,
    struct solver_options *solver_opts_sub,
    struct ESR_options
        *esr_opts_sub,  // create these outside because we use the same settings
                        // every time we solve an innner system
    struct NVESR_options* nvesr_opts_sub,
    struct SPMV_context *spmv_ctx_sub) {
  struct PCG_info *pcg_info_sub = create_pcg_info();

  // make sure the initial guess is set to zero
  gsl_vector_set_zero(vec);

  solve_linear_system(broken_comm, my_broken_index, num_broken_nodes, M_sub,
                      P_sub, rhs, vec, solver_opts_sub, esr_opts_sub,
                      nvesr_opts_sub, spmv_ctx_sub, pcg_info_sub);

  // check to make sure the system converged
  if (pcg_info_sub->converged_iterations)
    failure("Inner system did not converge");

  // free memory
  free_pcg_info(pcg_info_sub);
}
