// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "moe_gate_activation.cl"

// Batched GEMV decode of the fused MoE for u3 (3-bit unsigned) weights, on f16
// activations: one work-group per (token, top-k slot) pair and SGN * NB output
// columns. It reads the expert ids on the GPU and needs no gather or scatter.
//
// Weights are the plain [E, N, K] u3 tensor: each output channel is a
// LSB-first bit stream of K values, so a run of 32 values (a granule) is exactly
// 3 uints. Lane l owns granule l of every CHUNK-value run of K and loads its 32
// activations straight from the (cache resident) input, with no SLM staging or
// barrier, so the weight loads are issued right away.
//
// GEMV_DOWN computes the down projection of act(gate) * up, scaled by the routing
// weight; otherwise each subgroup computes NB gate and NB up columns and stores
// act(gate) * up.
//
// SHARED_EXPERT adds the shared expert as slot MAX_TOPK of every token, as in
// mlp_gate_up/mlp_down: its weights have the routed experts' layout with E = 1,
// and its routing weight is sigmoid(x . shared_gate_gate_weight), which the
// gate/up GEMV writes to shared_gate_out for the down GEMV.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define SIMD          16
#define K_CHUNK       32                    // u3 values per 3-uint granule
#define GRANULE_UINTS 3
#define CHUNKS_K      (K_SIZE / K_CHUNK)
#define W_GROUPS_K    (K_SIZE / WEI_GROUP_SIZE)
#define CHUNK         (SIMD * K_CHUNK)
#define NCHUNK        (K_SIZE / CHUNK)
#if (K_SIZE % CHUNK) != 0 || (WEI_GROUP_SIZE % K_CHUNK) != 0 || (N_SIZE % (SGN * NB)) != 0
#   error "moe_3gemm_int3.cl - invalid GEMV configuration"
#endif
#if GEMV_DOWN
#   define MATS 1
#else
#   define MATS 2
#endif
#if SHARED_EXPERT
#   define EXPERTS_PER_TOKEN (MAX_TOPK + 1)
#else
#   define EXPERTS_PER_TOKEN MAX_TOPK
#endif

// Eight consecutive u3 values (the low 24 bits of f; higher bits are ignored)
// spread into the eight nibbles of an int: into 12-bit halves at bits 0 and 16,
// then 6-bit quarters at each byte, then 3-bit values at each nibble. Each step
// keeps the low part in place and takes the rest from the shifted copy (one bfn);
// the bits the steps leave over are cleared at the end.
inline int FUNC(u3_spread8_nib)(uint f) {
    uint u = bitselect(f << 4, f, 0x00000FFFu);
    u = bitselect(u << 2, u, 0x003F003Fu);
    u = bitselect(u << 1, u, 0x07070707u);
    return as_int(u & 0x77777777u);
}

// Dot product of one granule with its 32 activations. Values i and i + 4 of each
// group of eight sit in the low and high half of a nibble pair; or-ing them into
// the mantissa of 1024.0 and subtracting 1024 gives them exactly as half2.
inline float FUNC(u3_dot32)(uint3 w, const half* a) {
    const uint f[4] = {w.s0, (w.s0 >> 24) | (w.s1 << 8), (w.s1 >> 16) | (w.s2 << 16), w.s2 >> 8};
    const half2 bias = (half2)((half)1024.0f);
    float acc = 0.0f;
    unroll_for (uint q = 0; q < 4; ++q) {
        const uint t = as_uint(FUNC_CALL(u3_spread8_nib)(f[q]));
        half2 s = (half2)((half)0.0f);
        unroll_for (uint i = 0; i < 4; ++i) {
            const half2 v = as_half2(((t >> (4 * i)) & 0x000F000Fu) | 0x64006400u) - bias;
            s = fma(v, (half2)(a[q * 8 + i], a[q * 8 + i + 4]), s);
        }
        acc += (float)s.s0 + (float)s.s1;
    }
    return acc;
}

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(SIMD, SGN, 1)))
KERNEL(moe_int3_gemv)(
    const __global int* expert_list,        // [tokens, MAX_TOPK]
    const __global uint* weights0,          // [E, N, K] u3: down, or gate
    const __global half* scale0,            // [E, W_GROUPS_K, N]
#if HAS_ZP
    const __global ZP_TYPE* zp0,            // [E, W_GROUPS_K, N] or [1]
#endif
#if !GEMV_DOWN
    const __global uint* weights1,          // up
    const __global half* scale1,
#if HAS_ZP
    const __global ZP_TYPE* zp1,
#endif
#endif
#if SHARED_EXPERT
    const __global uint* shared_weights0,   // [1, N, K] u3: down, or gate
    const __global half* shared_scale0,     // [1, W_GROUPS_K, N]
#if HAS_ZP
    const __global ZP_TYPE* shared_zp0,     // [1, W_GROUPS_K, N] or [1]
#endif
#if !GEMV_DOWN
    const __global uint* shared_weights1,   // up
    const __global half* shared_scale1,
#if HAS_ZP
    const __global ZP_TYPE* shared_zp1,
#endif
    const __global half* shared_gate_gate_weight,  // [K]
    __global half* shared_gate_out,                // [tokens]
#endif
#endif
    const __global half* x,                 // gate/up: [tokens, K]; down: [tokens * EXPERTS_PER_TOKEN, K]
#if GEMV_DOWN
    const __global half* routing_weights,   // [tokens * MAX_TOPK]
#if SHARED_EXPERT
    const __global half* shared_gate_in,    // [tokens]
#endif
#endif
    __global half* y)                       // [tokens * EXPERTS_PER_TOKEN, N]
{
    const uint lane = get_sub_group_local_id();
    const uint sg = (uint)get_local_id(1);
    const uint blocks = N_SIZE / (SGN * NB);
    const uint pair = (uint)get_group_id(2) / blocks;
    const uint block = (uint)get_group_id(2) % blocks;
    const uint n0 = (block * SGN + sg) * NB;
    const uint token = pair / EXPERTS_PER_TOKEN;
    const uint slot = pair % EXPERTS_PER_TOKEN;
#if SHARED_EXPERT
    const bool is_shared = slot == MAX_TOPK;
#else
    const bool is_shared = false;
#endif
    uint expert = 0;
    if (!is_shared)
        expert = (uint)expert_list[token * MAX_TOPK + slot];
#if GEMV_DOWN
    const __global half* xr = x + (size_t)pair * K_SIZE;
#else
    const __global half* xr = x + (size_t)token * K_SIZE;
#endif

    const __global uint* wm[MATS] = {weights0
#if !GEMV_DOWN
                                     , weights1
#endif
    };
    const __global half* sm[MATS] = {scale0
#if !GEMV_DOWN
                                     , scale1
#endif
    };
#if HAS_ZP
    const __global ZP_TYPE* zm[MATS] = {zp0
#if !GEMV_DOWN
                                        , zp1
#endif
    };
#endif
#if SHARED_EXPERT
    if (is_shared) {
        wm[0] = shared_weights0;
        sm[0] = shared_scale0;
#if HAS_ZP
        zm[0] = shared_zp0;
#endif
#if !GEMV_DOWN
        wm[1] = shared_weights1;
        sm[1] = shared_scale1;
#if HAS_ZP
        zm[1] = shared_zp1;
#endif
#endif
    }
#if !GEMV_DOWN
    // One subgroup per token computes the shared expert's routing weight
    if (is_shared && block == 0 && sg == 0) {
        float g = 0.0f;
        for (uint k = lane * 8; k < K_SIZE; k += SIMD * 8) {
            const half8 xv = vload8(0, xr + k);
            const half8 wv = vload8(0, shared_gate_gate_weight + k);
            unroll_for (uint t = 0; t < 8; ++t)
                g = fma((float)xv[t], (float)wv[t], g);
        }
        g = sub_group_reduce_add(g);
        if (lane == 0)
            shared_gate_out[token] = (half)(1.0f / (1.0f + exp(-g)));
    }
#endif
#endif

    float acc[MATS][NB];
    unroll_for (uint m = 0; m < MATS; ++m)
        unroll_for (uint j = 0; j < NB; ++j)
            acc[m][j] = 0.0f;

    unroll_for (uint c = 0; c < NCHUNK; ++c) {
        const uint gi = c * SIMD + lane;
        const uint wg = gi * K_CHUNK / WEI_GROUP_SIZE;
        half a[K_CHUNK];
        unroll_for (uint j = 0; j < K_CHUNK; j += 8) {
            const half8 v = vload8(0, xr + gi * K_CHUNK + j);
            unroll_for (uint t = 0; t < 8; ++t)
                a[j + t] = v[t];
        }
#if HAS_ZP
        // Activation sum of the granule, for the zero point
        float asum = 0.0f;
        unroll_for (uint j = 0; j < K_CHUNK; ++j)
            asum += (float)a[j];
#endif
        unroll_for (uint m = 0; m < MATS; ++m) {
            unroll_for (uint j = 0; j < NB; ++j) {
                const uint n = n0 + j;
                const uint3 w = vload3(gi, wm[m] + ((size_t)expert * N_SIZE + n) * CHUNKS_K * GRANULE_UINTS);
                float d = FUNC_CALL(u3_dot32)(w, a);
                const size_t sidx = ((size_t)expert * W_GROUPS_K + wg) * N_SIZE + n;
#if ZP_SCALAR
                d -= (float)zm[m][0] * asum;
#elif HAS_ZP
                d -= (float)zm[m][sidx] * asum;
#endif
                acc[m][j] = fma(d, (float)sm[m][sidx], acc[m][j]);
            }
        }
    }

#if GEMV_DOWN
    float routing_weight;
#if SHARED_EXPERT
    if (is_shared)
        routing_weight = (float)shared_gate_in[token];
    else
#endif
        routing_weight = (float)routing_weights[token * MAX_TOPK + slot];
#endif
    unroll_for (uint j = 0; j < NB; ++j) {
#if GEMV_DOWN
        const float out = sub_group_reduce_add(acc[0][j]) * routing_weight;
#else
        const float gate = sub_group_reduce_add(acc[0][j]);
        const float out = MOE_GATE_ACT(gate) * sub_group_reduce_add(acc[1][j]);
#endif
        if (lane == 0)
            y[(size_t)pair * N_SIZE + n0 + j] = (half)out;
    }
}

#undef SIMD
#undef K_CHUNK
#undef GRANULE_UINTS
#undef CHUNKS_K
#undef W_GROUPS_K
#undef CHUNK
#undef NCHUNK
#undef MATS
#undef EXPERTS_PER_TOKEN
