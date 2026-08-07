/*******************************************************************************
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// BatchGatherMatmul for u2 (2-bit unsigned, 4 values per byte, LSB-first) compressed weights.
// gemmstone micro-kernels and oneDNN have no u2 support, so u2 weights take this OCL path.
//
// Optimized coalesced-subgroup GEMV (replaces the 1-work-item-per-output reference): one
// subgroup computes one output channel n for one (token, expert_slot); the SG_SIZE lanes
// split the K reduction over packed weight uchar4s (16 u2 values per lane-load) so
// consecutive lanes read consecutive weight bytes (fully coalesced). The activation row
// is staged once per workgroup in SLM and shared by the CHANNELS_PER_WG channels of the
// (token, expert_slot), instead of being re-read from global memory per channel.
// Numerically identical to the reference: each byte holds 4 u2 values, is fully inside
// one quant group (group_size % 4 == 0), and is scaled/zp-corrected before the sub-group
// reduction.
//   Dispatch: local {SG_SIZE, CHANNELS_PER_WG, 1};
//             global {SG_SIZE, ceil(m / CHANNELS_PER_WG) * CHANNELS_PER_WG, n_tokens*top_k}.
//   dim1 = output channel n, dim2 = flat (token_idx, expert_slot).
// Scales/zp are physically [E, G, N] (bfyx when G == 1, byfx after prepare_quantization).

#include "include/batch_headers/sub_group_block_read.cl"

// Dot of the 4 u2 values packed in one byte with 4 SLM-staged activations.
inline float bgm_u2_dot4(const __local half* x4, uchar wb, float zp) {
    return convert_float(x4[0]) * (convert_float((wb >> 0) & 0x3) - zp)
         + convert_float(x4[1]) * (convert_float((wb >> 2) & 0x3) - zp)
         + convert_float(x4[2]) * (convert_float((wb >> 4) & 0x3) - zp)
         + convert_float(x4[3]) * (convert_float((wb >> 6) & 0x3) - zp);
}

#ifdef WEIGHT_ZP_DT
// Zero point for zp element zp_flat (= group * M_GEMM + channel); z is the per-expert zp base.
inline float bgm_u2_get_zp(
#    ifdef WEIGHT_ZP_SCALAR
    // Scalar (per-tensor) zp: one element shared by all experts/groups/channels.
    const __global WEIGHT_ZP_DT* z,
#    elif defined(WEIGHT_COMPRESSED_ZP_INT2) || defined(WEIGHT_COMPRESSED_ZP_INT4)
    const __global uchar* z,
#    else
    const __global WEIGHT_ZP_DT* z,
#    endif
    uint zp_flat) {
#    ifdef WEIGHT_ZP_SCALAR
    return convert_float(z[0]);
#    elif defined(WEIGHT_COMPRESSED_ZP_INT2)
    return convert_float((z[zp_flat / 4] >> ((zp_flat % 4) * 2)) & 0x3);
#    elif defined(WEIGHT_COMPRESSED_ZP_INT4)
    return convert_float((z[zp_flat / 2] >> ((zp_flat % 2) * 4)) & 0xF);
#    else
    return convert_float(z[zp_flat]);
#    endif
}
#endif

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(bgm_u2_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const global half* input_ptr,
    const global uchar* weight_ptr,
    global half* out_ptr,
    const global int* indices,
    int m,
    int k
#ifdef BIAS_DT
    , const global BIAS_DT* bias_ptr
#endif
    , const global WEIGHT_SCALE_DT* weight_scales
#ifdef WEIGHT_ZP_DT
    , const global WEIGHT_ZP_DT* weight_zps
#endif
) {
    uint flat_idx = get_global_id(2);
    uint top_k = TOP_K;
    uint token_idx = flat_idx / top_k;
    uint expert_slot = flat_idx % top_k;
    int n_tokens = N_TOKENS;

    uint n_act = N_ACTIVATED_EXPERTS;
    uint a_slot = min(expert_slot, n_act - 1);
    const global half* x = input_ptr + (a_slot * n_tokens + token_idx) * INPUT_STRIDE;

    // Stage the activation row in SLM once per workgroup; all CHANNELS_PER_WG channels
    // of this (token, expert_slot) then read it from SLM instead of global memory.
    __local half x_slm[K_GEMM];
    const uint linear = get_local_linear_id();
    for (uint i = linear; i < K_GEMM; i += SG_SIZE * CHANNELS_PER_WG)
        x_slm[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint n = get_global_id(1);
    if (n >= (uint)m)
        return;

    const uint lane = get_sub_group_local_id();
    int expert_id = indices[token_idx * top_k + expert_slot];

    // u2: 4 values per byte; EXPERT_STRIDE is the per-expert byte count (elements / 4).
    const global uchar* w = weight_ptr + (long)expert_id * EXPERT_STRIDE + (long)n * (K_GEMM / 4);

    const global WEIGHT_SCALE_DT* s = weight_scales + (long)expert_id * NUM_GROUPS * M_GEMM;
#ifdef WEIGHT_ZP_DT
#    ifdef WEIGHT_ZP_SCALAR
    const global WEIGHT_ZP_DT* z = weight_zps;
#    elif defined(WEIGHT_COMPRESSED_ZP_INT2)
    const global uchar* z = (const global uchar*)weight_zps + ((long)expert_id * NUM_GROUPS * M_GEMM) / 4;
#    elif defined(WEIGHT_COMPRESSED_ZP_INT4)
    const global uchar* z = (const global uchar*)weight_zps + ((long)expert_id * NUM_GROUPS * M_GEMM) / 2;
#    else
    const global WEIGHT_ZP_DT* z = weight_zps + (long)expert_id * NUM_GROUPS * M_GEMM;
#    endif
#endif

    const uint group_size = K_GEMM / NUM_GROUPS;
    const uint group_bytes = group_size / 4;  // group_size % 4 == 0 guaranteed by the generator
    const uint n_vec = K_GEMM / 16;           // K_GEMM % 16 == 0 guaranteed by the generator

    float acc = 0.0f;
    // Each lane reads 4 packed weight bytes (16 u2 values) per iteration; consecutive
    // lanes read consecutive uchar4s, so the hardware coalesces the per-lane vector
    // loads. Do NOT use a subgroup block read here: the base pointer is lane-dependent
    // (collectives require a uniform base), and a uniform-base variant would read past
    // the tile end and fault at the end of the mmap'd weights buffer.
    for (uint v = lane; v < n_vec; v += SG_SIZE) {
        uchar4 wb = vload4(v, (const __global uchar*)w);
#if GROUP_VEC_ALIGNED
        // group_size % 16 == 0: the whole uchar4 (16 values) sits inside one quant group.
        uint g = v / (group_bytes / 4);
        float scale = convert_float(s[g * M_GEMM + n]);
        float zp = 0.0f;
#    ifdef WEIGHT_ZP_DT
        zp = bgm_u2_get_zp(z, g * M_GEMM + n);
#    endif
        float d = bgm_u2_dot4(x_slm + v * 16 + 0, wb.s0, zp)
                + bgm_u2_dot4(x_slm + v * 16 + 4, wb.s1, zp)
                + bgm_u2_dot4(x_slm + v * 16 + 8, wb.s2, zp)
                + bgm_u2_dot4(x_slm + v * 16 + 12, wb.s3, zp);
        acc += d * scale;
#else
        // Narrow quant groups: scale/zp can change between the 4 packed bytes.
        unroll_for (uint j = 0; j < 4; ++j) {
            uint b = v * 4 + j;
            uint g = b / group_bytes;
            float scale = convert_float(s[g * M_GEMM + n]);
            float zp = 0.0f;
#    ifdef WEIGHT_ZP_DT
            zp = bgm_u2_get_zp(z, g * M_GEMM + n);
#    endif
            acc += bgm_u2_dot4(x_slm + b * 4, wb[j], zp) * scale;
        }
#endif
    }

    acc = sub_group_reduce_add(acc);
    if (lane == 0) {
#ifdef BIAS_DT
        acc += convert_float(bias_ptr[(long)expert_id * BIAS_STRIDE + n]);
#endif
        out_ptr[(expert_slot * n_tokens + token_idx) * OUTPUT_STRIDE + n] = convert_half(acc);
    }
}
