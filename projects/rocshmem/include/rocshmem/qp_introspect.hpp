/******************************************************************************
 * Per-peer GDA QP introspection for external (non-rocSHMEM) WQE builders.
 *
 * Exposes the raw mlx5 SQ ring / doorbell / BlueFlame / CQ device VAs, the live
 * SQ post counter address, and the local lkey / per-peer rkey / qpn for a peer's
 * RC QP, so a Triton (or other) kernel can post WQEs directly into a QP that
 * rocSHMEM already set up. Intentionally a flat POD + free function so callers
 * (e.g. the pybind layer) need no internal rocSHMEM headers.
 *
 * Only valid when the active backend is the GDA backend; returns false otherwise.
 *****************************************************************************/
#ifndef ROCSHMEM_QP_INTROSPECT_HPP
#define ROCSHMEM_QP_INTROSPECT_HPP

#include <cstdint>

namespace rocshmem {

struct QpInfoC {
  uint64_t sq_buf;     // SQ ring buffer device VA
  uint64_t sq_post;    // device VA of the live uint64 SQ post counter (continue it)
  uint64_t dbrec;      // SQ doorbell record device VA
  uint64_t bf;         // BlueFlame register device VA
  uint64_t cq_buf;     // CQ buffer device VA
  uint64_t base_heap;  // local symmetric heap base for this QP
  uint32_t sq_depth;   // SQ depth (WQEBBs)
  uint32_t cq_depth;   // CQ depth (CQEs); best-effort (mlx5: ~2*sq_depth)
  uint32_t lkey;       // local heap MR lkey
  uint32_t rkey;       // peer heap MR rkey (nic 0)
  uint32_t qpn;        // QP number
};

/// Fill *out for the RC QP to `peer`. Returns false if the active backend is not
/// the GDA backend (or `peer`/QP is unavailable). Collective-independent (host).
bool rocshmem_query_qp_info(int peer, int ctx_id, QpInfoC* out);

}  // namespace rocshmem

#endif  // ROCSHMEM_QP_INTROSPECT_HPP
