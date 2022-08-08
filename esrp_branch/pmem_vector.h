#ifndef PMEM_VECTOR_H_
#define PMEM_VECTOR_H_

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <libpmemobj.h>

POBJ_LAYOUT_BEGIN(pmem_vector_store);
POBJ_LAYOUT_TOID(pmem_vector_store, struct pmem_block);
POBJ_LAYOUT_TOID(pmem_vector_store, struct pmem_vector);
POBJ_LAYOUT_END(pmem_vector_store);

// ================= PMEM Block =================

struct pmem_block {
  size_t size;
  PMEMoid data;
};

TOID(struct pmem_block) pmem_block_alloc(PMEMobjpool* pool, size_t n);

int pmem_block_from_gsl_block(PMEMobjpool* pool, gsl_block* block,
                              TOID(struct pmem_block) * block_p);

int pmem_block_to_gsl_block(TOID(struct pmem_block) * block_p,
                            gsl_block* block);

void pmem_block_free(TOID(struct pmem_block) * block_p);

// ================= PMEM Vector =================

struct pmem_vector {
  size_t size;
  size_t stride;
  TOID(struct pmem_block) block;
};

TOID(struct pmem_vector) pmem_vector_alloc(PMEMobjpool* pool, size_t n);

int pmem_vector_from_gsl_vector(PMEMobjpool* pool, gsl_vector* vector,
                                TOID(struct pmem_vector) * vector_p);

int pmem_vector_to_gsl_vector(TOID(struct pmem_vector) * vector_p,
                              gsl_vector* vector);

void pmem_vector_free(TOID(struct pmem_vector) * vector_p);

#endif  // PMEM_VECTOR_H_
