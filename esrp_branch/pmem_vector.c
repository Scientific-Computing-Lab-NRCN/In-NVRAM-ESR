#include "pmem_vector.h"

#include <mpi.h>

#ifndef failure
#define failure(message)                     \
  do {                                       \
    perror(message);                         \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  } while (0)
#endif

TOID(struct pmem_block) pmem_block_alloc(PMEMobjpool* pool, size_t n) {
  struct pobj_action act[2];
  PMEMoid data = pmemobj_reserve(pool, &act[0], sizeof(double) * n, 0);
  if (OID_IS_NULL(data)) {
    failure("Failed to allocate blocks data:");
  }

  TOID(struct pmem_block) block_p;
  PMEMoid block_p_oid =
      pmemobj_reserve(pool, &act[1], sizeof(struct pmem_block), 1);

  TOID_ASSIGN(block_p, block_p_oid);
  if (TOID_IS_NULL(block_p)) {
    failure("Can't allocate block: ");
  }
  D_RW(block_p)->data = data;
  D_RW(block_p)->size = n;
  pmemobj_persist(pool, D_RW(block_p), sizeof(struct pmem_block));
  pmemobj_publish(pool, act, 2);
  return block_p;
}

int pmem_block_from_gsl_block(PMEMobjpool* pool, gsl_block* block,
                              TOID(struct pmem_block) * block_p) {
  size_t block_size = gsl_block_size(block);
  if (D_RO(*block_p)->size != block_size) {
    fprintf(stderr, "mismatch between pmem block and gsl_block sizes");
    return -1;
  }
  pmemobj_memcpy_persist(pool, pmemobj_direct(D_RW(*block_p)->data),
                         gsl_block_data(block), block_size * sizeof(double));
  return 0;
}

int pmem_block_to_gsl_block(TOID(struct pmem_block) * block_p,
                            gsl_block* block) {
  size_t block_size = gsl_block_size(block);
  if (D_RO(*block_p)->size != block_size) {
    fprintf(stderr, "mismatch between pmem block and gsl_block sizes");
    return -1;
  }
  memcpy(gsl_block_data(block), pmemobj_direct(D_RO(*block_p)->data),
         block_size * sizeof(double));
  return 0;
}

void pmem_block_free(TOID(struct pmem_block) * block_p) {
  pmemobj_free(&D_RW(*block_p)->data);
  pmemobj_free(block_p);
}

// ================= PMEM Vector =================

TOID(struct pmem_vector) pmem_vector_alloc(PMEMobjpool* pool, size_t n) {
  struct pobj_action act;

  TOID(struct pmem_vector) vector_p;
  PMEMoid vector_p_oid =
      pmemobj_reserve(pool, &act, sizeof(struct pmem_vector), 1);
  TOID_ASSIGN(vector_p, vector_p_oid);
  if (TOID_IS_NULL(vector_p)) {
    failure("Can't allocate block: ");
  }

  D_RW(vector_p)->block = pmem_block_alloc(pool, n);
  D_RW(vector_p)->size = n;
  D_RW(vector_p)->stride = 1;
  pmemobj_persist(pool, D_RW(vector_p), sizeof(struct pmem_vector));
  pmemobj_publish(pool, &act, 1);
  return vector_p;
}

int pmem_vector_from_gsl_vector(PMEMobjpool* pool, gsl_vector* vector,
                                TOID(struct pmem_vector) * vector_p) {
  size_t vector_size = vector->size;
  if (D_RO(*vector_p)->size != vector_size) {
    fprintf(stderr,
            "pmem_vector_from_gsl_vector: mismatch between pmem vector "
            "and gsl_vector sizes. pmem vector of size %ld, but got "
            "gsl_vector of size %ld.\n",
            D_RO(*vector_p)->size, vector_size);
    return -1;
  }
  pmem_block_from_gsl_block(pool, vector->block, &D_RW(*vector_p)->block);
  return 0;
}

int pmem_vector_to_gsl_vector(TOID(struct pmem_vector) * vector_p,
                              gsl_vector* vector) {
  size_t vector_size = vector->size;
  if (D_RO(*vector_p)->size != vector_size) {
    fprintf(stderr,
            "pmem_vector_to_gsl_vector: mismatch between pmem vector and "
            "gsl_vector sizes. pmem vector of size %ld, but got gsl_vector of "
            "size %ld.\n",
            D_RO(*vector_p)->size, vector_size);
    return -1;
  }
  pmem_block_to_gsl_block(&D_RW(*vector_p)->block, vector->block);
  return 0;
}

void pmem_vector_free(TOID(struct pmem_vector) * vector_p) {
  pmem_block_free(&D_RW(*vector_p)->block);
}
