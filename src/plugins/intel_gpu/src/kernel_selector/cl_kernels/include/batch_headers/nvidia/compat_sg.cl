// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// Sub-group emulation for NVIDIA: sub-group size / id queries, shuffles, barrier and voting, broadcast, reductions and
// block read / write, all parameterised on the emulated sub-group size.
//
// NVIDIA hardware executes in 32-wide warps, but OpenVINO kernels ask for sub-groups of 32, 16 or 8 through
// REQD_SUB_GROUP_SIZE(). A narrower sub-group is emulated by treating every aligned group of _NV_SUB_GROUP_SIZE lanes
// inside a warp as one sub-group: shuffles are confined to that segment and reductions stop at that width.
//
// Everything that depends on the sub-group size takes it as an explicit sg_size argument and is reached through a
// macro that passes _NV_SUB_GROUP_SIZE. Macro bodies are expanded at the call site, so redefining _NV_SUB_GROUP_SIZE
// ahead of a kernel changes the behaviour of that kernel alone. A plain function would instead bake in whatever value
// _NV_SUB_GROUP_SIZE had where the function was defined - and since the batch headers are emitted once at the top of
// the program, that value would always be the 32 default. This is why every helper below is parameterised.
//
// kernels_cache.cpp rewrites REQD_SUB_GROUP_SIZE(x) and __attribute__((intel_reqd_sub_group_size(x))) into a
// redefinition of _NV_SUB_GROUP_SIZE at the top of the owning source entry and restores the default at its end.
// nvidia/compat_sg_overrides.cl, emitted after the batch headers, redirects the names that those headers define as
// functions (_sub_group_block_read*, _sub_group_block_write*, _sub_group_shuffle) to the parameterised helpers here.
//
// kernels_cache.cpp emits this file as part of the NVIDIA prelude; nvidia_prelude_names there fixes the order.
// Dependencies are listed here instead of as #include lines: primitive_db_gen.py strips #include from batch headers
// anyway, and its dependency sort fails on include graphs that reach the same header along two paths.
//
// Depends on:
//     nvidia/compat_common.cl   cl_khr_fp16, _nv_get_linear_tid / _nv_get_total_threads (work-item ids)
//     nvidia/compat_half.cl     as_half / as_ushort, half max / min

#ifdef __NV_CL_C_VERSION

// Emulated sub-group size of a kernel that asks for none: a whole warp. kernels_cache.cpp restores it by this name
// at the end of each source entry, so the value lives here only.
#define _NV_DEFAULT_SUB_GROUP_SIZE 32u
// Emulated sub-group size in effect; kernels_cache.cpp redefines it per source entry (see the file header).
#define _NV_SUB_GROUP_SIZE _NV_DEFAULT_SUB_GROUP_SIZE

// --------------------------------------------------------------------------------------------------------------------
// Lane id
// --------------------------------------------------------------------------------------------------------------------
// The work-item ids (_nv_get_linear_tid, _nv_get_total_threads) come from compat_common.cl, which also binds
// get_local_linear_id() to them. %laneid does not change while a work-item runs either, so this asm is not volatile
// like them: the compiler may then fold the reads that every shuffle, vote and block access repeats through
// _nv_sg_mask and _nv_get_sub_group_local_id. The work-item id comment in compat_common.cl lists what that code
// shape exposed. The shfl.sync / vote.sync / bar.warp.sync asm further down stays volatile, since it synchronises
// the warp.

static inline uint _nv_laneid(void){
    uint lane;
    __asm__("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// --------------------------------------------------------------------------------------------------------------------
// Emulated sub-group membership and shfl.sync operands
// --------------------------------------------------------------------------------------------------------------------

// Work-items of the caller's emulated sub-group that the work-group actually launched. A sub-group is narrower than
// sg_size whenever the work-group is smaller than one sub-group - softmax_gpu_bf asks for a 16-wide sub-group and
// dispatches a local size of 4 - or when the work-group size leaves the last sub-group short.
static inline uint _nv_sg_valid_count(uint sg_size){
    const uint linear = _nv_get_linear_tid();
    const uint base = linear - (linear % sg_size);
    const uint remaining = _nv_get_total_threads() - base;
    return remaining < sg_size ? remaining : sg_size;
}

// Lanes of the current warp that make up the caller's emulated sub-group. Lanes the work-group never launched are
// left out: naming them in a shfl.sync / vote.sync / bar.warp.sync member mask is undefined behaviour, and a mask
// that claims they exist makes vote.sync.ballot based helpers such as sub_group_all read as if those lanes had
// voted false. This also covers the last warp of a work-group whose size is not a multiple of 32 on the
// non-emulated path, where sg_size is 32.
static inline uint _nv_sg_mask(uint sg_size){
    const uint base = _nv_laneid() & ~(sg_size - 1u);
    const uint valid = _nv_sg_valid_count(sg_size);
    const uint bits = (valid >= 32u) ? 0xffffffffu : ((1u << valid) - 1u);
    return bits << base;
}

// Clamp / segmentation operand of shfl.sync: bits [12:8] hold the segment mask that selects the lane bits identifying
// the segment, bits [4:0] the clamp value, with
//     maxLane = (lane & segmask) | (clamp & ~segmask)     minLane = lane & segmask
// This is the encoding CUDA's __shfl_sync / __shfl_down_sync / __shfl_xor_sync build for a given width.
static inline uint _nv_sg_clamp(uint sg_size){
    return ((32u - sg_size) << 8u) | 0x1fu;
}

// Same, for shfl.sync.up, where a clamp value of 0 makes maxLane the base lane of the segment.
static inline uint _nv_sg_clamp_up(uint sg_size){
    return (32u - sg_size) << 8u;
}

// --------------------------------------------------------------------------------------------------------------------
// Sub-group queries
// --------------------------------------------------------------------------------------------------------------------

static inline uint _nv_get_max_sub_group_size(uint sg_size){
    return sg_size;
}
#define get_max_sub_group_size() _nv_get_max_sub_group_size(_NV_SUB_GROUP_SIZE)

static inline uint _nv_get_sub_group_local_id(uint sg_size){
    return _nv_get_linear_tid() % sg_size;
}
#define get_sub_group_local_id() _nv_get_sub_group_local_id(_NV_SUB_GROUP_SIZE)

static inline uint _nv_get_sub_group_id(uint sg_size){
    return _nv_get_linear_tid() / sg_size;
}
#define get_sub_group_id() _nv_get_sub_group_id(_NV_SUB_GROUP_SIZE)

static inline uint _nv_get_sub_group_size(uint sg_size){
    uint nxyz = _nv_get_total_threads();
    uint remainder = nxyz % sg_size;
    if (remainder == 0) return sg_size;
    uint linear = _nv_get_linear_tid();
    return linear < (nxyz / sg_size) * sg_size ? sg_size : remainder;
}
#define get_sub_group_size() _nv_get_sub_group_size(_NV_SUB_GROUP_SIZE)

static inline uint _nv_get_num_sub_groups(uint sg_size){
    uint nxyz = _nv_get_total_threads();
    return (nxyz + (sg_size - 1)) / sg_size;
}
#define get_num_sub_groups() _nv_get_num_sub_groups(_NV_SUB_GROUP_SIZE)

// --------------------------------------------------------------------------------------------------------------------
// Raw 32-bit shuffles
// --------------------------------------------------------------------------------------------------------------------

static inline uint _nv_shfl_idx_u32(uint x, uint lane, uint sg_size){
    uint c = _nv_sg_clamp(sg_size);
    uint m = _nv_sg_mask(sg_size);
    uint r;
    __asm__ volatile("shfl.sync.idx.b32 %0,%1,%2,%3,%4;" : "=r"(r) : "r"(x), "r"(lane), "r"(c), "r"(m));
    return r;
}

static inline uint _nv_shfl_bfly_u32(uint x, uint lane_mask, uint sg_size){
    uint c = _nv_sg_clamp(sg_size);
    uint m = _nv_sg_mask(sg_size);
    uint r;
    __asm__ volatile("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;" : "=r"(r) : "r"(x), "r"(lane_mask), "r"(c), "r"(m));
    return r;
}

static inline uint _nv_shfl_up_u32(uint x, uint delta, uint sg_size){
    uint c = _nv_sg_clamp_up(sg_size);
    uint m = _nv_sg_mask(sg_size);
    uint r;
    __asm__ volatile("shfl.sync.up.b32 %0,%1,%2,%3,%4;" : "=r"(r) : "r"(x), "r"(delta), "r"(c), "r"(m));
    return r;
}

static inline uint _nv_shfl_down_u32(uint x, uint delta, uint sg_size){
    uint c = _nv_sg_clamp(sg_size);
    uint m = _nv_sg_mask(sg_size);
    uint r;
    __asm__ volatile("shfl.sync.down.b32 %0,%1,%2,%3,%4;" : "=r"(r) : "r"(x), "r"(delta), "r"(c), "r"(m));
    return r;
}

// --------------------------------------------------------------------------------------------------------------------
// Barrier and voting
// --------------------------------------------------------------------------------------------------------------------

static inline void _nv_sub_group_barrier(uint flags, uint sg_size){
    uint m = _nv_sg_mask(sg_size);
    __asm__ volatile("bar.warp.sync %0;" :: "r"(m) : "memory");
}
#define sub_group_barrier(flags) _nv_sub_group_barrier((flags), _NV_SUB_GROUP_SIZE)

static inline uint _nv_sg_ballot(int predicate, uint sg_size){
    uint m = _nv_sg_mask(sg_size);
    uint r;
    __asm__ volatile("{ .reg .pred %%p; setp.ne.s32 %%p, %1, 0; vote.sync.ballot.b32 %0, %%p, %2; }"
                     : "=r"(r) : "r"(predicate), "r"(m));
    return r;
}

static inline int _nv_sub_group_any(int predicate, uint sg_size){
    uint m = _nv_sg_mask(sg_size);
    return (_nv_sg_ballot(predicate, sg_size) & m) != 0u;
}
#define sub_group_any(p) _nv_sub_group_any((p), _NV_SUB_GROUP_SIZE)

static inline int _nv_sub_group_all(int predicate, uint sg_size){
    uint m = _nv_sg_mask(sg_size);
    return (_nv_sg_ballot(predicate, sg_size) & m) == m;
}
#define sub_group_all(p) _nv_sub_group_all((p), _NV_SUB_GROUP_SIZE)

// --------------------------------------------------------------------------------------------------------------------
// sub_group_broadcast
// --------------------------------------------------------------------------------------------------------------------

// Integer types up to 32 bits travel in one 32-bit shuffle. utype is the unsigned type of the same width: it
// zero-extends the value into the shuffled register and truncates it back, as in OV_NV_SGSHUD below.
#define OV_NV_SGBC(type, utype)                                                                               \
static inline type __attribute__((overloadable)) _nv_sub_group_broadcast(type x, uint lane, uint sg){         \
    return (type)(utype)_nv_shfl_idx_u32((uint)(utype)x, lane, sg);                                           \
}
// float and half travel as the bits of the unsigned integer of the same width.
#define OV_NV_SGBC_AS(type, btype)                                                                            \
static inline type __attribute__((overloadable)) _nv_sub_group_broadcast(type x, uint lane, uint sg){         \
    return as_##type(_nv_sub_group_broadcast(as_##btype(x), lane, sg));                                       \
}

OV_NV_SGBC(uint, uint)
OV_NV_SGBC(int, uint)
OV_NV_SGBC(ushort, ushort)
OV_NV_SGBC(short, ushort)
OV_NV_SGBC(uchar, uchar)
OV_NV_SGBC(char, uchar)
OV_NV_SGBC_AS(float, uint)
OV_NV_SGBC_AS(half, ushort)
#undef OV_NV_SGBC
#undef OV_NV_SGBC_AS

static inline ulong __attribute__((overloadable)) _nv_sub_group_broadcast(ulong x, uint lane, uint sg){
    uint lo = _nv_shfl_idx_u32((uint)(x & 0xffffffffUL), lane, sg);
    uint hi = _nv_shfl_idx_u32((uint)(x >> 32), lane, sg);
    return ((ulong)hi << 32) | (ulong)lo;
}
static inline long __attribute__((overloadable)) _nv_sub_group_broadcast(long x, uint lane, uint sg){
    return (long)_nv_sub_group_broadcast((ulong)x, lane, sg);
}
#define sub_group_broadcast(x, lane) _nv_sub_group_broadcast((x), (lane), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_broadcast(x, lane) _nv_sub_group_broadcast((x), (lane), _NV_SUB_GROUP_SIZE)

// --------------------------------------------------------------------------------------------------------------------
// sub-group shuffle
// --------------------------------------------------------------------------------------------------------------------
// intel_sub_group_shuffle allows a per-lane index, unlike sub_group_broadcast. shfl.sync.idx reads the index from a
// register per lane, so routing shuffle through the broadcast helper keeps the per-lane index intact.
// The _sub_group_shuffle* spellings are bound in nvidia/compat_sg_overrides.cl, after the batch headers have
// declared their own (sub-group-size 32) versions of those names.

#define OV_NV_SGSH_1(type)                                                                                    \
static inline type __attribute__((overloadable)) _nv_sub_group_shuffle(type v, uint c, uint sg){              \
    return _nv_sub_group_broadcast(v, c, sg);                                                                 \
}
#define OV_NV_SGSH_2(type)                                                                                    \
static inline type##2 __attribute__((overloadable)) _nv_sub_group_shuffle(type##2 v, uint c, uint sg){        \
    return (type##2)(_nv_sub_group_broadcast(v.s0, c, sg), _nv_sub_group_broadcast(v.s1, c, sg));             \
}
#define OV_NV_SGSH_4(type)                                                                                    \
static inline type##4 __attribute__((overloadable)) _nv_sub_group_shuffle(type##4 v, uint c, uint sg){        \
    return (type##4)(_nv_sub_group_broadcast(v.s0, c, sg), _nv_sub_group_broadcast(v.s1, c, sg),              \
                     _nv_sub_group_broadcast(v.s2, c, sg), _nv_sub_group_broadcast(v.s3, c, sg));             \
}
#define OV_NV_SGSH_8(type)                                                                                    \
static inline type##8 __attribute__((overloadable)) _nv_sub_group_shuffle(type##8 v, uint c, uint sg){        \
    return (type##8)(_nv_sub_group_broadcast(v.s0, c, sg), _nv_sub_group_broadcast(v.s1, c, sg),              \
                     _nv_sub_group_broadcast(v.s2, c, sg), _nv_sub_group_broadcast(v.s3, c, sg),              \
                     _nv_sub_group_broadcast(v.s4, c, sg), _nv_sub_group_broadcast(v.s5, c, sg),              \
                     _nv_sub_group_broadcast(v.s6, c, sg), _nv_sub_group_broadcast(v.s7, c, sg));             \
}
#define OV_NV_SGSH_16(type)                                                                                   \
static inline type##16 __attribute__((overloadable)) _nv_sub_group_shuffle(type##16 v, uint c, uint sg){      \
    return (type##16)(_nv_sub_group_broadcast(v.s0, c, sg), _nv_sub_group_broadcast(v.s1, c, sg),             \
                      _nv_sub_group_broadcast(v.s2, c, sg), _nv_sub_group_broadcast(v.s3, c, sg),             \
                      _nv_sub_group_broadcast(v.s4, c, sg), _nv_sub_group_broadcast(v.s5, c, sg),             \
                      _nv_sub_group_broadcast(v.s6, c, sg), _nv_sub_group_broadcast(v.s7, c, sg),             \
                      _nv_sub_group_broadcast(v.s8, c, sg), _nv_sub_group_broadcast(v.s9, c, sg),             \
                      _nv_sub_group_broadcast(v.sa, c, sg), _nv_sub_group_broadcast(v.sb, c, sg),             \
                      _nv_sub_group_broadcast(v.sc, c, sg), _nv_sub_group_broadcast(v.sd, c, sg),             \
                      _nv_sub_group_broadcast(v.se, c, sg), _nv_sub_group_broadcast(v.sf, c, sg));            \
}
#define OV_NV_SGSH(type) OV_NV_SGSH_1(type) OV_NV_SGSH_2(type) OV_NV_SGSH_4(type) OV_NV_SGSH_8(type) OV_NV_SGSH_16(type)

OV_NV_SGSH(int)
OV_NV_SGSH(uint)
OV_NV_SGSH(float)
OV_NV_SGSH(half)
OV_NV_SGSH(short)
OV_NV_SGSH(ushort)
OV_NV_SGSH(char)
OV_NV_SGSH(uchar)
OV_NV_SGSH_1(long)
OV_NV_SGSH_1(ulong)

#undef OV_NV_SGSH
#undef OV_NV_SGSH_1
#undef OV_NV_SGSH_2
#undef OV_NV_SGSH_4
#undef OV_NV_SGSH_8
#undef OV_NV_SGSH_16

#define intel_sub_group_shuffle(v, c) _nv_sub_group_shuffle((v), (c), _NV_SUB_GROUP_SIZE)

// intel_sub_group_shuffle_up(prev, cur, delta) logically concatenates prev (low half) and cur (high half) into a
// 2*sg element array and returns element (sg + id - delta); shuffle_down returns element (id + delta) of
// concat(cur, next). Both source shuffles execute on every lane so the sub-group stays converged.
#define OV_NV_SGSHUD(type, utype)                                                                                      \
static inline type __attribute__((overloadable)) _nv_sub_group_shuffle_up(type prev, type cur, uint delta, uint sg){   \
    uint id = _nv_get_sub_group_local_id(sg);                                                                          \
    utype from_cur  = (utype)_nv_shfl_up_u32((uint)(utype)cur, delta, sg);                                             \
    utype from_prev = (utype)_nv_shfl_down_u32((uint)(utype)prev, sg - delta, sg);                                     \
    return (type)((id >= delta) ? from_cur : from_prev);                                                               \
}                                                                                                                      \
static inline type __attribute__((overloadable)) _nv_sub_group_shuffle_down(type cur, type next, uint delta, uint sg){ \
    uint id = _nv_get_sub_group_local_id(sg);                                                                          \
    utype from_cur  = (utype)_nv_shfl_down_u32((uint)(utype)cur, delta, sg);                                           \
    utype from_next = (utype)_nv_shfl_up_u32((uint)(utype)next, sg - delta, sg);                                       \
    return (type)((id + delta < sg) ? from_cur : from_next);                                                           \
}

OV_NV_SGSHUD(uint, uint)
OV_NV_SGSHUD(int, uint)
OV_NV_SGSHUD(ushort, ushort)
OV_NV_SGSHUD(short, ushort)
OV_NV_SGSHUD(uchar, uchar)
OV_NV_SGSHUD(char, uchar)
#undef OV_NV_SGSHUD

// float and half shuffle as the bits of the unsigned integer of the same width.
#define OV_NV_SGSHUD_AS(type, btype)                                                                                   \
static inline type __attribute__((overloadable)) _nv_sub_group_shuffle_up(type prev, type cur, uint delta, uint sg){   \
    return as_##type(_nv_sub_group_shuffle_up(as_##btype(prev), as_##btype(cur), delta, sg));                          \
}                                                                                                                      \
static inline type __attribute__((overloadable)) _nv_sub_group_shuffle_down(type cur, type next, uint delta, uint sg){ \
    return as_##type(_nv_sub_group_shuffle_down(as_##btype(cur), as_##btype(next), delta, sg));                        \
}

OV_NV_SGSHUD_AS(float, uint)
OV_NV_SGSHUD_AS(half, ushort)
#undef OV_NV_SGSHUD_AS

#define intel_sub_group_shuffle_up(prev, cur, delta)   _nv_sub_group_shuffle_up((prev), (cur), (delta), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_shuffle_down(cur, next, delta) _nv_sub_group_shuffle_down((cur), (next), (delta), _NV_SUB_GROUP_SIZE)

// --------------------------------------------------------------------------------------------------------------------
// sub-group reductions
// --------------------------------------------------------------------------------------------------------------------

// The butterfly below is only correct while every lane it names exists, so it is kept for the full sub-group and a
// short one walks the work-items it does have instead. A partial butterfly is not an option: with a non power of two
// number of valid lanes an XOR exchange folds in values that already aggregate absent lanes, so no per-step guard
// recovers the right answer. The linear form costs O(n) shuffles, but it only runs on sub-groups the work-group left
// short; the full path is the one every existing kernel takes. Every lane of a short sub-group folds the values in
// the same order, lane 0 first, so all of them return the same bits. Starting from the lane's own value would let
// float and half sums differ in the last bit between lanes, which the butterfly never does: each of its exchanges
// combines the same two values on both lanes.
#define OV_NV_SG_REDUCE(name, type, combine, to_u, from_u)                                                             \
static inline type __attribute__((overloadable)) name(type val, uint sg){                                              \
    const uint n = _nv_sg_valid_count(sg);                                                                             \
    type r = val;                                                                                                      \
    if (n == sg) {                                                                                                     \
        for (uint off = sg / 2u; off > 0u; off /= 2u)                                                                  \
            r = combine(r, from_u(_nv_shfl_bfly_u32(to_u(r), off, sg)));                                               \
    } else {                                                                                                           \
        r = from_u(_nv_shfl_idx_u32(to_u(val), 0u, sg));                                                               \
        for (uint j = 1u; j < n; ++j)                                                                                  \
            r = combine(r, from_u(_nv_shfl_idx_u32(to_u(val), j, sg)));                                                \
    }                                                                                                                  \
    return r;                                                                                                          \
}

#define OV_NV_SG_KEEP_U(v)   (v)
#define OV_NV_SG_U_OF_INT(v) ((uint)(v))
#define OV_NV_SG_INT_OF_U(v) ((int)(v))
#define OV_NV_SG_U_OF_H(v)   ((uint)as_ushort(v))
#define OV_NV_SG_H_OF_U(v)   as_half((ushort)(v))

// half arithmetic goes through float, matching the accumulation the non-emulated built-ins perform.
#define OV_NV_SG_ADD(a, b)   ((a) + (b))
#define OV_NV_SG_ADD_H(a, b) ((half)((float)(a) + (float)(b)))
#define OV_NV_SG_MAX(a, b)   max((a), (b))
#define OV_NV_SG_MAX_F(a, b) fmax((a), (b))
#define OV_NV_SG_MIN(a, b)   min((a), (b))
#define OV_NV_SG_MIN_F(a, b) fmin((a), (b))

OV_NV_SG_REDUCE(_nv_sub_group_reduce_add, uint,  OV_NV_SG_ADD,   OV_NV_SG_KEEP_U,   OV_NV_SG_KEEP_U)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_add, int,   OV_NV_SG_ADD,   OV_NV_SG_U_OF_INT, OV_NV_SG_INT_OF_U)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_add, float, OV_NV_SG_ADD,   as_uint,           as_float)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_add, half,  OV_NV_SG_ADD_H, OV_NV_SG_U_OF_H,   OV_NV_SG_H_OF_U)
#define sub_group_reduce_add(v) _nv_sub_group_reduce_add((v), _NV_SUB_GROUP_SIZE)

OV_NV_SG_REDUCE(_nv_sub_group_reduce_max, uint,  OV_NV_SG_MAX,   OV_NV_SG_KEEP_U,   OV_NV_SG_KEEP_U)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_max, int,   OV_NV_SG_MAX,   OV_NV_SG_U_OF_INT, OV_NV_SG_INT_OF_U)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_max, float, OV_NV_SG_MAX_F, as_uint,           as_float)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_max, half,  OV_NV_SG_MAX,   OV_NV_SG_U_OF_H,   OV_NV_SG_H_OF_U)
#define sub_group_reduce_max(v) _nv_sub_group_reduce_max((v), _NV_SUB_GROUP_SIZE)

OV_NV_SG_REDUCE(_nv_sub_group_reduce_min, uint,  OV_NV_SG_MIN,   OV_NV_SG_KEEP_U,   OV_NV_SG_KEEP_U)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_min, int,   OV_NV_SG_MIN,   OV_NV_SG_U_OF_INT, OV_NV_SG_INT_OF_U)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_min, float, OV_NV_SG_MIN_F, as_uint,           as_float)
OV_NV_SG_REDUCE(_nv_sub_group_reduce_min, half,  OV_NV_SG_MIN,   OV_NV_SG_U_OF_H,   OV_NV_SG_H_OF_U)
#define sub_group_reduce_min(v) _nv_sub_group_reduce_min((v), _NV_SUB_GROUP_SIZE)

#undef OV_NV_SG_REDUCE
#undef OV_NV_SG_KEEP_U
#undef OV_NV_SG_U_OF_INT
#undef OV_NV_SG_INT_OF_U
#undef OV_NV_SG_U_OF_H
#undef OV_NV_SG_H_OF_U
#undef OV_NV_SG_ADD
#undef OV_NV_SG_ADD_H
#undef OV_NV_SG_MAX
#undef OV_NV_SG_MAX_F
#undef OV_NV_SG_MIN
#undef OV_NV_SG_MIN_F

// --------------------------------------------------------------------------------------------------------------------
// sub-group block read / write
// --------------------------------------------------------------------------------------------------------------------
// Lane i of a sub-group accesses elements i, i + sg, i + 2*sg ... from the base pointer shared by the sub-group.
// The address space is part of the signature, so one name serves both the plain and the _slm family through the
// __global and __local overloads.
//
// The base pointer is taken from the sub-group's first lane. cl_intel_subgroups requires it to be the same for
// every work-item, and Intel hardware enforces that by reading the address out of a uniform register, so a kernel
// that hands the built-in a pointer varying across the sub-group still gets the first lane's. Several kernels rely
// on this: sdpa_opt.cl folds head_size_idx, which is get_local_id(2), into the value pointer and then expects the
// block read to supply the per-lane part, so reading through each lane's own pointer would add it twice. The
// accessors below hand lane 0's pointer to every lane through the ulong overload of _nv_sub_group_broadcast.

#define OV_NV_CAT_(a, b) a##b
#define OV_NV_CAT(a, b) OV_NV_CAT_(a, b)
#define OV_NV_VEC_1(t)  t
#define OV_NV_VEC_2(t)  t##2
#define OV_NV_VEC_4(t)  t##4
#define OV_NV_VEC_8(t)  t##8
#define OV_NV_VEC_16(t) t##16
#define OV_NV_VEC(t, w) OV_NV_CAT(OV_NV_VEC_, w)(t)

#define OV_NV_SGBR_1   r    = p[i];
#define OV_NV_SGBR_2   r.s0 = p[i]; i += sg; r.s1 = p[i]; i += sg;
#define OV_NV_SGBR_4   OV_NV_SGBR_2 r.s2 = p[i]; i += sg; r.s3 = p[i]; i += sg;
#define OV_NV_SGBR_8   OV_NV_SGBR_4 r.s4 = p[i]; i += sg; r.s5 = p[i]; i += sg; r.s6 = p[i]; i += sg; r.s7 = p[i]; i += sg;
#define OV_NV_SGBR_16  OV_NV_SGBR_8 r.s8 = p[i]; i += sg; r.s9 = p[i]; i += sg; r.sa = p[i]; i += sg; r.sb = p[i]; i += sg; \
                                    r.sc = p[i]; i += sg; r.sd = p[i]; i += sg; r.se = p[i]; i += sg; r.sf = p[i]; i += sg;

#define OV_NV_SGBW_1   p[i] = v;
#define OV_NV_SGBW_2   p[i] = v.s0; i += sg; p[i] = v.s1; i += sg;
#define OV_NV_SGBW_4   OV_NV_SGBW_2 p[i] = v.s2; i += sg; p[i] = v.s3; i += sg;
#define OV_NV_SGBW_8   OV_NV_SGBW_4 p[i] = v.s4; i += sg; p[i] = v.s5; i += sg; p[i] = v.s6; i += sg; p[i] = v.s7; i += sg;
#define OV_NV_SGBW_16  OV_NV_SGBW_8 p[i] = v.s8; i += sg; p[i] = v.s9; i += sg; p[i] = v.sa; i += sg; p[i] = v.sb; i += sg; \
                                    p[i] = v.sc; i += sg; p[i] = v.sd; i += sg; p[i] = v.se; i += sg; p[i] = v.sf; i += sg;

#define OV_NV_DECL_SGBR_AS(fname, type, w, aspace)                                                            \
static inline OV_NV_VEC(type, w) __attribute__((overloadable))                                                \
fname(const aspace type* base, uint sg_size){                                                                 \
    uint sg = sg_size;                                                                                        \
    uint i = _nv_get_sub_group_local_id(sg);                                                                  \
    const aspace type* p = (const aspace type*)(size_t)_nv_sub_group_broadcast((ulong)(size_t)base, 0u, sg);  \
    OV_NV_VEC(type, w) r;                                                                                     \
    OV_NV_CAT(OV_NV_SGBR_, w)                                                                                 \
    return r;                                                                                                 \
}

#define OV_NV_DECL_SGBW_AS(fname, type, w, aspace)                                                            \
static inline void __attribute__((overloadable))                                                              \
fname(aspace type* base, OV_NV_VEC(type, w) v, uint sg_size){                                                 \
    uint sg = sg_size;                                                                                        \
    uint i = _nv_get_sub_group_local_id(sg);                                                                  \
    aspace type* p = (aspace type*)(size_t)_nv_sub_group_broadcast((ulong)(size_t)base, 0u, sg);              \
    OV_NV_CAT(OV_NV_SGBW_, w)                                                                                 \
}

#define OV_NV_DECL_SGBR(fname, type, w) OV_NV_DECL_SGBR_AS(fname, type, w, __global) \
                                        OV_NV_DECL_SGBR_AS(fname, type, w, __local)
#define OV_NV_DECL_SGBW(fname, type, w) OV_NV_DECL_SGBW_AS(fname, type, w, __global) \
                                        OV_NV_DECL_SGBW_AS(fname, type, w, __local)

OV_NV_DECL_SGBR(_nv_sub_group_block_read,      uint,   1)
OV_NV_DECL_SGBR(_nv_sub_group_block_read2,     uint,   2)
OV_NV_DECL_SGBR(_nv_sub_group_block_read4,     uint,   4)
OV_NV_DECL_SGBR(_nv_sub_group_block_read8,     uint,   8)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_us,   ushort, 1)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_us2,  ushort, 2)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_us4,  ushort, 4)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_us8,  ushort, 8)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_uc,   uchar,  1)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_uc2,  uchar,  2)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_uc4,  uchar,  4)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_uc8,  uchar,  8)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_uc16, uchar,  16)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_ul,   ulong,  1)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_ul2,  ulong,  2)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_ul4,  ulong,  4)
OV_NV_DECL_SGBR(_nv_sub_group_block_read_ul8,  ulong,  8)

OV_NV_DECL_SGBW(_nv_sub_group_block_write,      uint,   1)
OV_NV_DECL_SGBW(_nv_sub_group_block_write2,     uint,   2)
OV_NV_DECL_SGBW(_nv_sub_group_block_write4,     uint,   4)
OV_NV_DECL_SGBW(_nv_sub_group_block_write8,     uint,   8)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_us,   ushort, 1)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_us2,  ushort, 2)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_us4,  ushort, 4)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_us8,  ushort, 8)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_uc,   uchar,  1)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_uc2,  uchar,  2)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_uc4,  uchar,  4)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_uc8,  uchar,  8)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_uc16, uchar,  16)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_ul,   ulong,  1)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_ul2,  ulong,  2)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_ul4,  ulong,  4)
OV_NV_DECL_SGBW(_nv_sub_group_block_write_ul8,  ulong,  8)

#undef OV_NV_DECL_SGBR
#undef OV_NV_DECL_SGBW
#undef OV_NV_DECL_SGBR_AS
#undef OV_NV_DECL_SGBW_AS
#undef OV_NV_SGBR_1
#undef OV_NV_SGBR_2
#undef OV_NV_SGBR_4
#undef OV_NV_SGBR_8
#undef OV_NV_SGBR_16
#undef OV_NV_SGBW_1
#undef OV_NV_SGBW_2
#undef OV_NV_SGBW_4
#undef OV_NV_SGBW_8
#undef OV_NV_SGBW_16
#undef OV_NV_VEC
#undef OV_NV_VEC_1
#undef OV_NV_VEC_2
#undef OV_NV_VEC_4
#undef OV_NV_VEC_8
#undef OV_NV_VEC_16
#undef OV_NV_CAT
#undef OV_NV_CAT_

// The intel_* spellings are used directly by several ocl_v2 kernels. Defining them here is safe because no batch
// header declares a function under these names on this target.
#define intel_sub_group_block_read(ptr)       _nv_sub_group_block_read((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read2(ptr)      _nv_sub_group_block_read2((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read4(ptr)      _nv_sub_group_block_read4((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read8(ptr)      _nv_sub_group_block_read8((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_us(ptr)    _nv_sub_group_block_read_us((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_us2(ptr)   _nv_sub_group_block_read_us2((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_us4(ptr)   _nv_sub_group_block_read_us4((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_us8(ptr)   _nv_sub_group_block_read_us8((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_uc(ptr)    _nv_sub_group_block_read_uc((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_uc2(ptr)   _nv_sub_group_block_read_uc2((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_uc4(ptr)   _nv_sub_group_block_read_uc4((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_uc8(ptr)   _nv_sub_group_block_read_uc8((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_uc16(ptr)  _nv_sub_group_block_read_uc16((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_ul(ptr)    _nv_sub_group_block_read_ul((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_ul2(ptr)   _nv_sub_group_block_read_ul2((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_ul4(ptr)   _nv_sub_group_block_read_ul4((ptr), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_read_ul8(ptr)   _nv_sub_group_block_read_ul8((ptr), _NV_SUB_GROUP_SIZE)

#define intel_sub_group_block_write(ptr, v)      _nv_sub_group_block_write((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write2(ptr, v)     _nv_sub_group_block_write2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write4(ptr, v)     _nv_sub_group_block_write4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write8(ptr, v)     _nv_sub_group_block_write8((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_us(ptr, v)   _nv_sub_group_block_write_us((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_us2(ptr, v)  _nv_sub_group_block_write_us2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_us4(ptr, v)  _nv_sub_group_block_write_us4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_us8(ptr, v)  _nv_sub_group_block_write_us8((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_uc(ptr, v)   _nv_sub_group_block_write_uc((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_uc2(ptr, v)  _nv_sub_group_block_write_uc2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_uc4(ptr, v)  _nv_sub_group_block_write_uc4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_uc8(ptr, v)  _nv_sub_group_block_write_uc8((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_uc16(ptr, v) _nv_sub_group_block_write_uc16((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_ul(ptr, v)   _nv_sub_group_block_write_ul((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_ul2(ptr, v)  _nv_sub_group_block_write_ul2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_ul4(ptr, v)  _nv_sub_group_block_write_ul4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define intel_sub_group_block_write_ul8(ptr, v)  _nv_sub_group_block_write_ul8((ptr), (v), _NV_SUB_GROUP_SIZE)

#endif // __NV_CL_C_VERSION
