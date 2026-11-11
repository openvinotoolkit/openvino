// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// Work-group collectives for the native NVIDIA OpenCL compiler: work_group_reduce_add / max / min,
// work_group_scan_inclusive_add and work_group_broadcast for float, and half overloads of the six collectives the
// kernels call (work_group_scan_exclusive_add included, whose float version comes from the runtime).
//
// kernels_cache.cpp adds this file to the NVIDIA prelude of every batch. The 128 bytes of static shared memory
// declared below (one float per warp of a 1024-wide work-group) live at PTX module scope, but ptxas allocates a
// module-scope .shared variable only to the kernels that reference it, directly or through the functions they call,
// so kernels that call none of the collectives carry none of it.
//
// The runtime declares the float collectives itself, without static, so the float definitions here are plain inline:
// static inline would clash with those declarations, and that non-inline declaration makes the inline definition an
// external one, so the kernels always run these. The half overloads have no runtime declaration, so they are static
// inline: a plain inline definition alone emits no function, which fails to link wherever a call is not inlined
// (as with -cl-opt-disable). As OpenCL requires, every work-item of the work-group has to call a collective; each
// call ends with a barrier, so consecutive calls can reuse the shared buffer.
//
// Depends on (both emitted ahead of this file as part of the NVIDIA prelude):
//     nvidia/compat_common.cl   _nv_get_linear_tid / _nv_get_total_threads (work-item ids)
//     nvidia/compat_sg.cl       _nv_laneid

#ifdef __NV_CL_C_VERSION

__asm__(".shared .align 4 .b8 __ov_wg_collective_smem[128];");

static inline uint __ov_nv_wg_smem_base(void){
    uint r;
    __asm__ volatile("mov.u32 %0,__ov_wg_collective_smem;" : "=r"(r));
    return r;
}

// work_group_reduce_add / max / min (float) in two phases: a butterfly inside each warp, then one over the per-warp
// results in __ov_wg_collective_smem. A lane folds in its partner only when the partner exists, so a short last warp
// or a work-group of fewer than 32 work-items never mixes in the value shfl.sync leaves undefined for absent lanes;
// lane 0, the only lane whose result is used, still sees every lane that exists.
// The runtime supplies reduce_max / reduce_min as well, but on sm_80 (driver 590.48.01) they fold a 0 into the
// result: reduce_max of all-negative inputs returns 0 at every work-group size, and reduce_min of all-positive inputs
// returns 0 whenever the work-group size is not a multiple of 32. softmax_gpu_bf subtracts the row maximum before
// exp(), so a row of logits around -200 comes out as NaN. Defining all three here takes precedence over the runtime.
#define OV_NV_WG_REDUCE_F(name, combine)                                                                              \
inline float __attribute__((overloadable)) name(float x){                                                             \
    const uint total = _nv_get_total_threads();                                                                       \
    const uint wid = _nv_get_linear_tid() >> 5u;                                                                      \
    const uint laneid = _nv_laneid();                                                                                 \
    uint wn = total - wid * 32u;                                                                                      \
    if (wn > 32u) wn = 32u;                                                                                           \
    const uint mask = (wn >= 32u) ? 0xffffffffu : ((1u << wn) - 1u);                                                  \
    float v = x;                                                                                                      \
    for (uint off = 16u; off > 0u; off >>= 1u) {                                                                      \
        uint t;                                                                                                       \
        __asm__ volatile("shfl.sync.bfly.b32 %0,%1,%2,31,%3;" : "=r"(t) : "r"(as_uint(v)), "r"(off), "r"(mask));      \
        if ((laneid ^ off) < wn) v = combine(v, as_float(t));                                                         \
    }                                                                                                                 \
    const uint smem = __ov_nv_wg_smem_base();                                                                         \
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
// per-warp offsets in __ov_wg_collective_smem, and every work-item adds its warp's offset to its in-warp result.
inline float __attribute__((overloadable)) work_group_scan_inclusive_add(float x){
    uint total = _nv_get_total_threads();
    uint fid = _nv_get_linear_tid();
    uint wid = fid >> 5u;
    uint laneid = _nv_laneid();
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
    uint smem = __ov_nv_wg_smem_base();
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
    uint fid = _nv_get_linear_tid();
    uint smem = __ov_nv_wg_smem_base();
    if (fid == (uint)id) {
        __asm__ volatile("st.shared.f32 [%0],%1;" :: "r"(smem), "f"(x));
    }
    __asm__ volatile("bar.sync 0;");
    float r;
    __asm__ volatile("ld.shared.f32 %0,[%1];" : "=f"(r) : "r"(smem));
    __asm__ volatile("bar.sync 0;");
    return r;
}

// The NVIDIA runtime declares the work-group collectives for int, uint, long, ulong, float and double, but not for
// half. A half argument converts to float and to double equally well, so every call with a half operand is ambiguous:
//
//   error: call to 'work_group_reduce_max' is ambiguous
//
// softmax_gpu_bf hits this whenever INPUT0_TYPE is half and the work-group is wider than a sub-group. Exact overloads
// resolve it, because an exact match beats any conversion. Reducing in float is no less accurate than accumulating in
// half. The float versions they forward to are defined above, except scan_exclusive_add, which the runtime supplies
// and which returns correct results on sm_80 for work-group sizes from 1 to 1024, including sizes that are not a
// multiple of the warp width and 2D / 3D work-groups.
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
