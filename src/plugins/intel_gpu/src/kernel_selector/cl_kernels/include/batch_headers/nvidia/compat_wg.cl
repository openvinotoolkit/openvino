// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// Work-group collectives for the native NVIDIA OpenCL compiler: work_group_reduce_add / max / min,
// work_group_scan_inclusive_add and work_group_broadcast for float, and half overloads of the six collectives the
// kernels call (work_group_scan_exclusive_add included, whose float version comes from the runtime).
//
// Every batch carries this file, but ptxas allocates the 128 bytes of shared memory below (one float per warp of a
// 1024-wide work-group) only to the kernels that reach a collective.
//
// The runtime declares the float collectives itself, without static, so the float definitions here are plain inline:
// static inline would clash with those declarations, and that non-inline declaration makes the inline definition an
// external one, so the kernels always run these. The half overloads have no runtime declaration, so they are static
// inline: a plain inline definition alone emits no function, which fails to link wherever a call is not inlined
// (as with -cl-opt-disable). As OpenCL requires, every work-item of the work-group has to call a collective; each
// call ends with a barrier, so consecutive calls can reuse the shared buffer.
//
// Depends on:
//     nvidia/compat_common.cl   ov_nv_get_linear_tid / ov_nv_get_total_threads (work-item ids)
//     nvidia/compat_sg.cl       ov_nv_laneid

#ifdef __NV_CL_C_VERSION

__asm__(".shared .align 4 .b8 ov_nv_wg_collective_smem[128];");

static inline uint ov_nv_wg_smem_base(void){
    uint r;
    __asm__ volatile("mov.u32 %0,ov_nv_wg_collective_smem;" : "=r"(r));
    return r;
}

// work_group_reduce_add / max / min (float): a butterfly in each warp, then one over the per-warp results in
// ov_nv_wg_collective_smem. A lane folds in its partner only when the partner exists, so short warps stay correct.
// The runtime's reduce_max / reduce_min fold a 0 into the result on sm_80 (driver 590.48.01), so all three are
// defined here.
#define OV_NV_WG_REDUCE_F(name, combine)                                                                              \
inline float __attribute__((overloadable)) name(float x){                                                             \
    const uint total = ov_nv_get_total_threads();                                                                     \
    const uint wid = ov_nv_get_linear_tid() >> 5u;                                                                    \
    const uint laneid = ov_nv_laneid();                                                                               \
    uint wn = total - wid * 32u;                                                                                      \
    if (wn > 32u) wn = 32u;                                                                                           \
    const uint mask = (wn >= 32u) ? 0xffffffffu : ((1u << wn) - 1u);                                                  \
    float v = x;                                                                                                      \
    for (uint off = 16u; off > 0u; off >>= 1u) {                                                                      \
        uint t;                                                                                                       \
        __asm__ volatile("shfl.sync.bfly.b32 %0,%1,%2,31,%3;" : "=r"(t) : "r"(as_uint(v)), "r"(off), "r"(mask));      \
        if ((laneid ^ off) < wn) v = combine(v, as_float(t));                                                         \
    }                                                                                                                 \
    const uint smem = ov_nv_wg_smem_base();                                                                           \
    if (laneid == 0u) { __asm__ volatile("st.shared.f32 [%0],%1;" :: "r"(smem + wid * 4u), "f"(v)); }                 \
    __asm__ volatile("bar.sync 0;");                                                                                  \
    if (wid == 0u) {                                                                                                  \
        const uint nw = (total + 31u) >> 5u;                                                                          \
        const uint mask0 = (total >= 32u) ? 0xffffffffu : ((1u << total) - 1u);                                       \
        float w = 0.0f;                                                                                               \
        if (laneid < nw) { __asm__ volatile("ld.shared.f32 %0,[%1];" : "=f"(w) : "r"(smem + laneid * 4u)); }          \
        for (uint off = 16u; off > 0u; off >>= 1u) {                                                                  \
            uint t;                                                                                                   \
            __asm__ volatile("shfl.sync.bfly.b32 %0,%1,%2,31,%3;" : "=r"(t) : "r"(as_uint(w)), "r"(off), "r"(mask0)); \
            if ((laneid ^ off) < nw) w = combine(w, as_float(t));                                                     \
        }                                                                                                             \
        if (laneid == 0u) { __asm__ volatile("st.shared.f32 [%0],%1;" :: "r"(smem), "f"(w)); }                        \
    }                                                                                                                 \
    __asm__ volatile("bar.sync 0;");                                                                                  \
    float r;                                                                                                          \
    __asm__ volatile("ld.shared.f32 %0,[%1];" : "=f"(r) : "r"(smem));                                                 \
    __asm__ volatile("bar.sync 0;");                                                                                  \
    return r;                                                                                                         \
}
#define OV_NV_WG_ADD(a, b) ((a) + (b))
OV_NV_WG_REDUCE_F(work_group_reduce_add, OV_NV_WG_ADD)
OV_NV_WG_REDUCE_F(work_group_reduce_max, fmax)
OV_NV_WG_REDUCE_F(work_group_reduce_min, fmin)
#undef OV_NV_WG_ADD
#undef OV_NV_WG_REDUCE_F

// work_group_scan_inclusive_add (float): an inclusive scan inside each warp; warp 0 then scans the warp totals into
// per-warp offsets in ov_nv_wg_collective_smem, and every work-item adds its warp's offset to its in-warp result.
inline float __attribute__((overloadable)) work_group_scan_inclusive_add(float x){
    uint total = ov_nv_get_total_threads();
    uint fid = ov_nv_get_linear_tid();
    uint wid = fid >> 5u;
    uint laneid = ov_nv_laneid();
    uint wnthreads = total - wid * 32u;
    if (wnthreads > 32u) wnthreads = 32u;
    uint mask = (wnthreads >= 32u) ? 0xffffffffu : ((1u << wnthreads) - 1u);
    uint v = as_uint(x), t;
    __asm__ volatile("shfl.sync.up.b32 %0,%1,1,0,%2;" : "=r"(t) : "r"(v), "r"(mask));
    if (laneid >= 1u) v = as_uint(as_float(v) + as_float(t));
    __asm__ volatile("shfl.sync.up.b32 %0,%1,2,0,%2;" : "=r"(t) : "r"(v), "r"(mask));
    if (laneid >= 2u) v = as_uint(as_float(v) + as_float(t));
    __asm__ volatile("shfl.sync.up.b32 %0,%1,4,0,%2;" : "=r"(t) : "r"(v), "r"(mask));
    if (laneid >= 4u) v = as_uint(as_float(v) + as_float(t));
    __asm__ volatile("shfl.sync.up.b32 %0,%1,8,0,%2;" : "=r"(t) : "r"(v), "r"(mask));
    if (laneid >= 8u) v = as_uint(as_float(v) + as_float(t));
    __asm__ volatile("shfl.sync.up.b32 %0,%1,16,0,%2;" : "=r"(t) : "r"(v), "r"(mask));
    if (laneid >= 16u) v = as_uint(as_float(v) + as_float(t));
    uint smem = ov_nv_wg_smem_base();
    uint nw = (total + 31u) >> 5u;
    if (laneid == wnthreads - 1u) {
        __asm__ volatile("st.shared.f32 [%0],%1;" :: "r"(smem + wid * 4u), "f"(as_float(v)));
    }
    __asm__ volatile("bar.sync 0;");
    if (wid == 0u) {
        uint mask0 = (total >= 32u) ? 0xffffffffu : ((1u << total) - 1u);
        float wt = 0.0f;
        if (laneid < nw) {
            __asm__ volatile("ld.shared.f32 %0,[%1];" : "=f"(wt) : "r"(smem + laneid * 4u));
        }
        uint wv = as_uint(wt);
        __asm__ volatile("shfl.sync.up.b32 %0,%1,1,0,%2;" : "=r"(t) : "r"(wv), "r"(mask0));
        if (laneid >= 1u) wv = as_uint(as_float(wv) + as_float(t));
        __asm__ volatile("shfl.sync.up.b32 %0,%1,2,0,%2;" : "=r"(t) : "r"(wv), "r"(mask0));
        if (laneid >= 2u) wv = as_uint(as_float(wv) + as_float(t));
        __asm__ volatile("shfl.sync.up.b32 %0,%1,4,0,%2;" : "=r"(t) : "r"(wv), "r"(mask0));
        if (laneid >= 4u) wv = as_uint(as_float(wv) + as_float(t));
        __asm__ volatile("shfl.sync.up.b32 %0,%1,8,0,%2;" : "=r"(t) : "r"(wv), "r"(mask0));
        if (laneid >= 8u) wv = as_uint(as_float(wv) + as_float(t));
        __asm__ volatile("shfl.sync.up.b32 %0,%1,16,0,%2;" : "=r"(t) : "r"(wv), "r"(mask0));
        if (laneid >= 16u) wv = as_uint(as_float(wv) + as_float(t));
        uint excl;
        __asm__ volatile("shfl.sync.up.b32 %0,%1,1,0,%2;" : "=r"(excl) : "r"(wv), "r"(mask0));
        if (laneid == 0u) excl = 0u;
        if (laneid < nw) {
            __asm__ volatile("st.shared.f32 [%0],%1;" :: "r"(smem + laneid * 4u), "f"(as_float(excl)));
        }
    }
    __asm__ volatile("bar.sync 0;");
    float off;
    __asm__ volatile("ld.shared.f32 %0,[%1];" : "=f"(off) : "r"(smem + wid * 4u));
    __asm__ volatile("bar.sync 0;");
    return as_float(v) + off;
}

// work_group_broadcast (float): the work-item whose flat local id equals id stores x, and every work-item reads it
// back. The one-index form is the one OpenCL defines for 1D work-groups.
inline float __attribute__((overloadable)) work_group_broadcast(float x, size_t id){
    uint fid = ov_nv_get_linear_tid();
    uint smem = ov_nv_wg_smem_base();
    if (fid == (uint)id) {
        __asm__ volatile("st.shared.f32 [%0],%1;" :: "r"(smem), "f"(x));
    }
    __asm__ volatile("bar.sync 0;");
    float r;
    __asm__ volatile("ld.shared.f32 %0,[%1];" : "=f"(r) : "r"(smem));
    __asm__ volatile("bar.sync 0;");
    return r;
}

// The runtime declares the collectives for int, uint, long, ulong, float and double but not half, so a half argument
// is ambiguous between float and double; these exact overloads resolve that and reduce in float. The float
// scan_exclusive_add they forward to is the runtime's, which is correct on sm_80.
static inline half __attribute__((overloadable)) work_group_reduce_add(half x){
    return (half)work_group_reduce_add((float)x);
}
static inline half __attribute__((overloadable)) work_group_reduce_max(half x){
    return (half)work_group_reduce_max((float)x);
}
static inline half __attribute__((overloadable)) work_group_reduce_min(half x){
    return (half)work_group_reduce_min((float)x);
}
static inline half __attribute__((overloadable)) work_group_scan_inclusive_add(half x){
    return (half)work_group_scan_inclusive_add((float)x);
}
static inline half __attribute__((overloadable)) work_group_scan_exclusive_add(half x){
    return (half)work_group_scan_exclusive_add((float)x);
}
static inline half __attribute__((overloadable)) work_group_broadcast(half x, size_t id){
    return (half)work_group_broadcast((float)x, id);
}

#endif // __NV_CL_C_VERSION
