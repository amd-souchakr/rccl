#include "common_kernel.h"
#include "common.h"


#define NCCL_SPINS_BEFORE_CHECK_ABORT 10000


__device__ void
generic_barrier(int nthreads, uint64_t &barrier_next, uint64_t* barriers)
{
  const int wid = threadIdx.x%WARP_SIZE;
  if (wid == 0) {
    barrier_next += nthreads / WARP_SIZE;
    __hip_atomic_fetch_add(barriers, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
    int spins = 0;
    int rate_limit = 50;
    while (__hip_atomic_load(barriers, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) << barrier_next) {
      spins++;
      if (spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
        if (__atomic_load_n(ncclShmem.comm.abortFlag, __ATOMIC_SEQ_CST)) {
          ncclShmem.aborted = 1;
          break;
        }
        spins = 0;
      }
      if (spins == 0 && rate_limit > 0) {
        rate_limit--;
        traceData(__LINE__, threadIdx.x, __hip_atomic_load((barriers), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP), barrier_next);
      }
      __builtin_amdgcn_s_sleep(1);
    }
    __asm__ __volatile__("s_wakeup");
  }
}
