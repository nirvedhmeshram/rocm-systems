/******************************************************************************
 * Per-peer GDA QP introspection (see include/rocshmem/qp_introspect.hpp).
 *
 * Surfaces the raw mlx5 RC QP resources rocSHMEM's GDA backend already set up so
 * an external WQE builder (e.g. a Triton kernel) can post into the same QP.
 * Lives in the library TU where the GDA internals are in scope; the public
 * surface is just the flat POD + free function.
 *****************************************************************************/
#include "rocshmem/qp_introspect.hpp"

#include <hip/hip_runtime.h>

#include "backend_bc.hpp"
#include "backend_type.hpp"
#include "gda/backend_gda.hpp"
#include "gda/queue_pair.hpp"
#include "gda/mlx5/provider_gda_mlx5.hpp"

namespace rocshmem {

extern Backend *backend;

bool GDABackend::fill_qp_info(int peer, QpInfoC *out) {
  if (out == nullptr) return false;
  if (peer < 0 || peer >= static_cast<int>(num_pes)) return false;
  if (host_qps == nullptr || gpu_qps == nullptr || heap_rkey == nullptr) return false;

  // One QP per (peer, qp_slot); take the first slot for this peer.
  const int conn = peer * static_cast<int>(num_qps_per_pe);

  // The QP field VALUES are written directly into gpu_qps[conn] (device memory)
  // by initialize_gpu_qp; host_qps[conn] is just the empty initial construct.
  // Copy the device struct back to host to read the device-VA values. The live
  // SQ post counter ADDRESS is taken from the device array itself.
  // QueuePair has no default ctor; copy the device bytes into raw aligned storage
  // and view them (read POD members only — no ctor/dtor runs).
  alignas(QueuePair) unsigned char buf[sizeof(QueuePair)];
  if (hipMemcpy(buf, &gpu_qps[conn], sizeof(QueuePair), hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  const QueuePair *gq = reinterpret_cast<const QueuePair *>(buf);

  out->sq_buf    = reinterpret_cast<uint64_t>(gq->mlx5_sq.buf);
  out->sq_post   = reinterpret_cast<uint64_t>(&gpu_qps[conn].mlx5_sq.post);  // live counter (device VA)
  out->dbrec     = reinterpret_cast<uint64_t>(gq->mlx5_sq.dbrec);
  out->bf        = reinterpret_cast<uint64_t>(gq->mlx5_sq.db);
  out->cq_buf    = reinterpret_cast<uint64_t>(gq->mlx5_cq.buf);
  out->base_heap = static_cast<uint64_t>(gq->base_heap);
  out->sq_depth  = static_cast<uint32_t>(gq->mlx5_sq.depth);
  // mlx5 GDA CQ is created collapsed (cqe=1 + IGNORE_OVERRUN): all completions
  // land at slot 0 with wqe_counter as the high-water mark. Report depth 1 so an
  // external drain reads slot 0.
  out->cq_depth  = 1;
  out->lkey      = gq->lkey;
  out->rkey      = heap_rkey[peer * num_nics_];                    // nic 0
  out->qpn       = gq->qp_num;
  return true;
}

bool rocshmem_query_qp_info(int peer, QpInfoC *out) {
  if (backend == nullptr) return false;
  if (backend->get_type() != BackendType::GDA_BACKEND) return false;
  return static_cast<GDABackend *>(backend)->fill_qp_info(peer, out);
}

}  // namespace rocshmem
