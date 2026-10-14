
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"

// Fake group size for compatibility and computation performance balance.
// Each gk-iteration of the inner GEMV loop processes FAKE_GROUP_SIZE K-elements
// using a single (scale, zp) entry, so FAKE_GROUP_SIZE must divide both
// GATE_UP_GROUP_SIZE and DOWN_GROUP_SIZE. The traditional value is 128 which
// matches the sub_group_block_read tile widths below; for models whose weight
// quantization uses a smaller group (e.g. 64) we shrink FAKE_GROUP_SIZE so the
// per-iteration accumulation stays within one quant group.
#if defined(GATE_UP_GROUP_SIZE) && defined(DOWN_GROUP_SIZE)
#    if GATE_UP_GROUP_SIZE < DOWN_GROUP_SIZE
#        define MOE_MIN_GROUP_SIZE GATE_UP_GROUP_SIZE
#    else
#        define MOE_MIN_GROUP_SIZE DOWN_GROUP_SIZE
#    endif
#    if MOE_MIN_GROUP_SIZE < 128
#        define FAKE_GROUP_SIZE MOE_MIN_GROUP_SIZE
#    else
#        define FAKE_GROUP_SIZE 128
#    endif
#else
#    define FAKE_GROUP_SIZE 128
#endif

// Number of K-elements each work-item handles per gk-iteration via the
// intel_sub_group_block_read tile. Drives the inner-loop variant selection.
// Supported values: 1 (u2 kernels only: FAKE_GROUP_SIZE = SUBGROUP_SIZE, e.g. SG=32 + FAKE=32),
// 2 (FAKE_GROUP_SIZE = 1 * SUBGROUP_SIZE * 2, e.g. SG=16 + 32),
// 4 (e.g. SG=32 + FAKE=128 or SG=16 + FAKE=64), and 8 (SG=16 + FAKE=128).
#define ELEMS_PER_LANE (FAKE_GROUP_SIZE / SUBGROUP_SIZE)

// Experts per token: MAX_TOPK for non-shared, MAX_TOPK+1 with shared expert.
// Used by batched GEMV to decompose flat workgroup ID into (token_idx, expert_slot).
#if SHARED_EXPERT_ENABLE
#define EXPERTS_PER_TOKEN (MAX_TOPK + 1)
#else
#define EXPERTS_PER_TOKEN MAX_TOPK
#endif

// Gate activation: SwiGLU (Swish, default), GeGLU-Tanh, or GeGLU-ERF.
// GATE_ACT_GELU_ERF takes precedence over GATE_ACT_GELU_TANH when both are set.
#ifdef GATE_ACT_GELU_ERF
// ERF Gelu: 0.5 * x * (1 + erf(x / sqrt(2))); A&S 7.1.26 fast erf approximation
inline float moe_mlp_fast_erf(float x) {
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float z = fabs(x);
    float t = 1.0f / (1.0f + p * z);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-(z * z));
    return (x >= 0.0f) ? y : -y;
}
#    define MOE_GATE_ACT(x) (0.5f * (x) * (1.0f + moe_mlp_fast_erf((x) * 0.7071067811865475f)))
#elif defined(GATE_ACT_GELU_TANH)
#    define MOE_GATE_ACT(x) (0.5f * (x) * (1.0f + (tanh(0.79788458347320556640625f * (x) * (1.0f + 0.044715f * (x) * (x))))))
#else
#    define MOE_GATE_ACT(x) ((x) / (1.0f + exp(-(x))))
#endif

// HAS_ZP: 1 = asymmetric quantization (subtract zero point), 0 = symmetric (no zero point)
#if HAS_ZP
#    define ZP_ADJUST_2(sum0, sum1, xg_sum_gk, z0, z1) ((sum0) - (xg_sum_gk) * (z0)), ((sum1) - (xg_sum_gk) * (z1))
#    define ZP_ADJUST_4(sum0123, xg_sum_gk, z) ((sum0123) - (xg_sum_gk) * (z))
#else
#    define ZP_ADJUST_2(sum0, sum1, xg_sum_gk, z0, z1) (sum0), (sum1)
#    define ZP_ADJUST_4(sum0123, xg_sum_gk, z) (sum0123)
#endif

// WEIGHT_IS_SIGNED: sign extension helpers for i4/i8 weights
#if WEIGHT_IS_SIGNED
// 4-bit sign extension: [0,15] -> [-8,7]. If bit3 is set, OR with 0xF0 to sign-extend to char.
#    define DEQUANT_4BIT_LO(v) convert_half((char)(((v) & 0x08) ? ((v) | 0xF0) : ((v) & 0x0F)))
#    define DEQUANT_4BIT_HI(v) convert_half((char)((((v) >> 4) & 0x08) ? (((v) >> 4) | 0xF0) : (((v) >> 4) & 0x0F)))
// 8-bit sign extension: reinterpret uchar as signed char
#    define DEQUANT_8BIT(v)    convert_half(as_char(v))
#else
#    define DEQUANT_4BIT_LO(v) convert_half((v) & 0x0F)
#    define DEQUANT_4BIT_HI(v) convert_half((v) >> 4)
#    define DEQUANT_8BIT(v)    convert_half(v)
#endif

// 2-bit (u2) dequant: 4 values per byte, LSB-first; u2 weights are always unsigned.
#define DEQUANT_2BIT(v, s) convert_half(((v) >> (s)) & 0x3)

// Group-scale load. MOE_SCALE_DT is half normally, uchar when the model stores an 8-bit group scale
// plus one per-tensor factor per GEMM (NNCF --experts-fp8-scale): 0.25 -> 0.125 bit/weight, which is
// -1.13 GB on the 160e 122B. Table lookup rather than open-coded bit surgery because e4m3's
// subnormals (below 2^-6) are where the near-dead groups land -- a bug there would be invisible in
// any aggregate error metric. Every e4m3 value is exactly representable in half (max 448 < 65504,
// min subnormal 2^-9 > half's 2^-24), so the table is exact, and the per-group factor costs one
// multiply per GROUP, not per element. Without f8 the factor is ignored entirely (the host asserts it
// is 1.0 then), so the f16 path keeps exactly the instructions it had.
#if MOE_SCALE_F8E4M3
__constant half MOE_SCALE_F8_LUT[256] = MOE_SCALE_F8_TABLE;
#    define MOE_SCALE_AT(S, idx, glob) (MOE_SCALE_F8_LUT[(uchar)((S)[(idx)])] * (half)(glob))
// One scale per lane (lane == output channel), as a float. The width MUST follow the dtype: the f16
// form reads 2 bytes per lane, so leaving it on an f8 buffer would silently pair up two e4m3 bytes
// into a half and stride twice as far.
#    define MOE_SCALE_BLOCK_READ(S, glob)                                                            \
        (convert_float(MOE_SCALE_F8_LUT[intel_sub_group_block_read_uc((const __global uchar*)(S))]) * \
         (float)(glob))
#else
#    define MOE_SCALE_AT(S, idx, glob) ((S)[(idx)])
#    define MOE_SCALE_BLOCK_READ(S, glob) \
        convert_float(as_half(intel_sub_group_block_read_us((const __global ushort*)(S))))
#endif

// A3 per-GEMM helpers (mixed-dtype MoE): per-expert byte strides given a compression
// code DT (0=u4/i4, 1=u8/i8, 2=f16, 3=u2) and group size GS. N*K equals
// INTERMEDIATE_SIZE*HIDDEN_SIZE for all three GEMMs, so one product serves gate/up/down.
#define MOE_WEI_PROD (INTERMEDIATE_SIZE * HIDDEN_SIZE)
#define MOE_EXPERT_WEI_BYTES(DT) ((DT) == 0 ? (MOE_WEI_PROD) / 2 : (DT) == 3 ? (MOE_WEI_PROD) / 4 : (MOE_WEI_PROD))
#define MOE_EXPERT_ZP_BYTES(DT, GS) \
    ((DT) == 0 ? (MOE_WEI_PROD) / 2 / (GS) : (DT) == 3 ? (MOE_WEI_PROD) / 4 / (GS) : (MOE_WEI_PROD) / (GS))

#if defined(U2_UNPACK_ENABLE)

// Unpacks u2-packed data (4 values per byte, LSB-first) into u4-packed data
// (2 values per byte, LSB-first), doubling the byte size while preserving the
// logical element order. Used to feed u2 MoE expert weights/zp to the u4-only
// prefill GEMM paths (micro-gemm/oneDNN have no u2 dtype).
inline uint moe_unpack_u2_byte(uint b) {
    // byte (v3 v2 v1 v0) -> uint16 ((v3 | v2 << 4) << 8) | (v0 | v1 << 4)
    return (b & 0x3u) | ((b & 0xCu) << 2) | ((b & 0x30u) << 4) | ((b & 0xC0u) << 6);
}

// broadcast_zp == 0: src holds src_count uints of u2-packed data; each work item
//                    unpacks one uint (4 bytes) into 2 uints (8 bytes) of u4 data.
// broadcast_zp == 1: src holds a single scalar zp element (byte); each work item
//                    replicates it into both nibbles of one output byte, so a
//                    per-tensor zp is materialized as a full u4 zp tensor and
//                    src_count is the output byte count.
// broadcast_zp == 2: src holds src_count single-byte zp VALUES (u8/i8, e.g. a
//                    per-channel zp); each work item packs two adjacent values
//                    into one u4 output byte, so dst holds ceil(src_count/2) bytes.
KERNEL(moe_unpack_u2_to_u4)(const __global uchar* src, __global uchar* dst, int src_count, int broadcast_zp) {
    const size_t i = get_global_id(0);
    if (broadcast_zp == 2) {
        if (i >= ((size_t)src_count + 1) / 2)
            return;
        const uint v0 = src[2 * i];
        const uint v1 = (2 * i + 1 < (size_t)src_count) ? src[2 * i + 1] : 0;
        dst[i] = (uchar)((v0 & 0xFu) | ((v1 & 0xFu) << 4));
        return;
    }
    if (i >= (size_t)src_count)
        return;
    if (broadcast_zp != 0) {
        const uint v = src[0];
        dst[i] = (uchar)(v | (v << 4));
        return;
    }
    const uint v = ((const __global uint*)src)[i];
    ((__global uint*)dst)[2 * i] = moe_unpack_u2_byte(v & 0xFFu) | (moe_unpack_u2_byte((v >> 8) & 0xFFu) << 16);
    ((__global uint*)dst)[2 * i + 1] = moe_unpack_u2_byte((v >> 16) & 0xFFu) | (moe_unpack_u2_byte(v >> 24) << 16);
}

#elif GATE_UP_ENABLE
inline void gate_up_gemv_n2x_u4(const __global uchar* weight,
                                __global MOE_SCALE_DT* scales,
                                half scale_global,
                                __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                half* x2,
                                float* xg_sum,
                                const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global MOE_SCALE_DT* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n / 2;
#endif
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half s0 = MOE_SCALE_AT(S, scale_offset, scale_global);
            half s1 = MOE_SCALE_AT(S, scale_offset + 1, scale_global);
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N / 2;
            uchar z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);
#endif

#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (1 byte = 2 nibbles).
            half sum0;
            half sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0 = fma(a.s0, (DEQUANT_4BIT_LO(b)), (half)0);
            sum0 = fma(a.s1, (DEQUANT_4BIT_HI(b)), sum0);

            sum1 = fma(a.s0, (DEQUANT_4BIT_LO(b2)), (half)0);
            sum1 = fma(a.s1, (DEQUANT_4BIT_HI(b2)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b.s2)), 0);
            sum0.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b.s3)), 0);

            sum0.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);
            sum0.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b.s2)), sum0.s2);
            sum0.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b.s3)), sum0.s3);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b2.s2)), 0);
            sum1.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b2.s3)), 0);

            sum1.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);
            sum1.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b2.s2)), sum1.s2);
            sum1.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b2.s3)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

// zp_scalar must come from the caller, not from a preprocessor branch: one compiled body serves
// both the gate and the up projection, and a mixed INT2_SYM/INT2_ASYM layer gives them different
// zero-point forms. Both callers pass a jit constant, so the selects below fold away.
inline void gate_up_gemv_n2x_u2(const __global uchar* weight,
                                __global MOE_SCALE_DT* scales,
                                half scale_global,
                                __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                half* x2,
                                float* xg_sum,
                                const bool silu,
                                const bool zp_scalar) {
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 4;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global MOE_SCALE_DT* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n / 4;
        // n is even, so channels n and n+1 always share one packed zp byte.
        const int zshift = (n & 3) * 2;
#endif
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half s0 = MOE_SCALE_AT(S, scale_offset, scale_global);
            half s1 = MOE_SCALE_AT(S, scale_offset + 1, scale_global);
#if HAS_ZP
            const half z_scalar = convert_half(((__global MOE_ZP_SCALAR_DT*)zps)[0]);
            const uchar z = zp_scalar ? (uchar)0 : Z[(gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N / 4];
            const half z_hf0 = zp_scalar ? z_scalar : convert_half((z >> zshift) & 0x3);
            const half z_hf1 = zp_scalar ? z_scalar : convert_half((z >> (zshift + 2)) & 0x3);
#endif

#    if ELEMS_PER_LANE == 4
            // Each lane reads 4 K-elements (1 byte = 4 x 2-bit values).
            half2 sum0;
            half2 sum1;
            half4 a = vload4(id_local, x2 + gk * FAKE_GROUP_SIZE);
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 4);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 4) + gk * FAKE_GROUP_SIZE / 4));

            sum0.s0 = fma(a.s0, (DEQUANT_2BIT(b, 0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_2BIT(b, 2)), 0);
            sum0.s0 = fma(a.s2, (DEQUANT_2BIT(b, 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (DEQUANT_2BIT(b, 6)), sum0.s1);

            sum1.s0 = fma(a.s0, (DEQUANT_2BIT(b2, 0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_2BIT(b2, 2)), 0);
            sum1.s0 = fma(a.s2, (DEQUANT_2BIT(b2, 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (DEQUANT_2BIT(b2, 6)), sum1.s1);

#    if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#    endif
#    elif ELEMS_PER_LANE == 2
            // Each lane handles 2 K-elements; two lanes share one packed byte.
            // NOTE: keep per-lane scalar loads here. A subgroup block read would make
            // lanes >= FAKE_GROUP_SIZE/4 read past the tile, and at the end of the
            // mmap'd weights buffer that overhang faults (CL_OUT_OF_RESOURCES).
            half sum0;
            half sum1;
            half2 a = vload2(id_local, x2 + gk * FAKE_GROUP_SIZE);
            uchar b = B[gk * FAKE_GROUP_SIZE / 4 + (id_local >> 1)];
            uchar b2 = B[(K / 4) + gk * FAKE_GROUP_SIZE / 4 + (id_local >> 1)];
            const int wshift = (id_local & 1) * 4;

            sum0 = fma(a.s0, (DEQUANT_2BIT(b, wshift)), (half)0);
            sum0 = fma(a.s1, (DEQUANT_2BIT(b, wshift + 2)), sum0);

            sum1 = fma(a.s0, (DEQUANT_2BIT(b2, wshift)), (half)0);
            sum1 = fma(a.s1, (DEQUANT_2BIT(b2, wshift + 2)), sum1);

#    if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#    endif
#    elif ELEMS_PER_LANE == 1
            // Each lane handles 1 K-element; four lanes share one packed byte.
            // Same per-lane scalar loads as the ==2 branch: subgroup block reads
            // would overhang the weights buffer and fault (CL_OUT_OF_RESOURCES).
            half sum0;
            half sum1;
            half a = x2[gk * FAKE_GROUP_SIZE + id_local];
            uchar b = B[gk * FAKE_GROUP_SIZE / 4 + (id_local >> 2)];
            uchar b2 = B[(K / 4) + gk * FAKE_GROUP_SIZE / 4 + (id_local >> 2)];
            const int wshift = (id_local & 3) * 2;

            sum0 = fma(a, (DEQUANT_2BIT(b, wshift)), (half)0);
            sum1 = fma(a, (DEQUANT_2BIT(b2, wshift)), (half)0);

#    if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#    endif
#    else
            half4 sum0;
            half4 sum1;
            // The block read is strided: lane L gets byte L and byte L+SUBGROUP_SIZE, which cover
            // K elements 4L.. and 4L+FAKE_GROUP_SIZE/2.. respectively. The activations must be
            // loaded in two matching contiguous quads, not as one vload8 of 8L...
            half4 alo = vload4(id_local, x2 + gk * FAKE_GROUP_SIZE);
            half4 ahi = vload4(id_local, x2 + gk * FAKE_GROUP_SIZE + FAKE_GROUP_SIZE / 2);
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 4);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 4) + gk * FAKE_GROUP_SIZE / 4));

            sum0.s0 = fma(alo.s0, (DEQUANT_2BIT(b.s0, 0)), 0);
            sum0.s1 = fma(alo.s1, (DEQUANT_2BIT(b.s0, 2)), 0);
            sum0.s2 = fma(alo.s2, (DEQUANT_2BIT(b.s0, 4)), 0);
            sum0.s3 = fma(alo.s3, (DEQUANT_2BIT(b.s0, 6)), 0);

            sum0.s0 = fma(ahi.s0, (DEQUANT_2BIT(b.s1, 0)), sum0.s0);
            sum0.s1 = fma(ahi.s1, (DEQUANT_2BIT(b.s1, 2)), sum0.s1);
            sum0.s2 = fma(ahi.s2, (DEQUANT_2BIT(b.s1, 4)), sum0.s2);
            sum0.s3 = fma(ahi.s3, (DEQUANT_2BIT(b.s1, 6)), sum0.s3);

            sum1.s0 = fma(alo.s0, (DEQUANT_2BIT(b2.s0, 0)), 0);
            sum1.s1 = fma(alo.s1, (DEQUANT_2BIT(b2.s0, 2)), 0);
            sum1.s2 = fma(alo.s2, (DEQUANT_2BIT(b2.s0, 4)), 0);
            sum1.s3 = fma(alo.s3, (DEQUANT_2BIT(b2.s0, 6)), 0);

            sum1.s0 = fma(ahi.s0, (DEQUANT_2BIT(b2.s1, 0)), sum1.s0);
            sum1.s1 = fma(ahi.s1, (DEQUANT_2BIT(b2.s1, 2)), sum1.s1);
            sum1.s2 = fma(ahi.s2, (DEQUANT_2BIT(b2.s1, 4)), sum1.s2);
            sum1.s3 = fma(ahi.s3, (DEQUANT_2BIT(b2.s1, 6)), sum1.s3);

#    if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#    endif
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

inline void gate_up_gemv_n2x_u8(const __global uchar* weight,
                                __global MOE_SCALE_DT* scales,
                                half scale_global,
                                __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                half* x2,
                                float* xg_sum,
                                const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global MOE_SCALE_DT* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n;
#endif
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half s0 = MOE_SCALE_AT(S, scale_offset, scale_global);
            half s1 = MOE_SCALE_AT(S, scale_offset + 1, scale_global);
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half z0 = convert_half(Z[zp_offset]);
            half z1 = convert_half(Z[zp_offset + 1]);
#endif

#    if ELEMS_PER_LANE == 4
            float2 sum0;
            float2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), sum0.s0);
            sum0.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), sum0.s1);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), sum1.s0);
            sum1.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (8-bit weights, 1 byte each).
            float sum0;
            float sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), sum0);

            sum1 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            float4 sum0;
            float4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), 0.0f);
            sum0.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), 0.0f);

            sum0.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b.s4)), sum0.s0);
            sum0.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b.s5)), sum0.s1);
            sum0.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b.s6)), sum0.s2);
            sum0.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b.s7)), sum0.s3);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), 0.0f);
            sum1.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), 0.0f);

            sum1.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b2.s4)), sum1.s0);
            sum1.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b2.s5)), sum1.s1);
            sum1.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b2.s6)), sum1.s2);
            sum1.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b2.s7)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

inline void gate_up_gemv_n2x_f16(const __global half* weight, __global half* y, int N, int K, half* x2, const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global half* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s0 = fma(a.s2, b.s2, sum0.s0);
            sum0.s1 = fma(a.s3, b.s3, sum0.s1);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
            sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

            sum_all0 += sum0[0] + sum0[1];
            sum_all1 += sum1[0] + sum1[1];
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 fp16 elements.
            half sum0;
            half sum1;
            half2 a = as_half2(intel_sub_group_block_read_us2((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half2 b = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half2 b2 = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0 = fma(a.s0, b.s0, (half)0);
            sum0 = fma(a.s1, b.s1, sum0);

            sum1 = fma(a.s0, b2.s0, (half)0);
            sum1 = fma(a.s1, b2.s1, sum1);

            sum_all0 += sum0;
            sum_all1 += sum1;
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half8 b2 = as_half8(intel_sub_group_block_read_us8((const __global ushort*)(B + K + gk * FAKE_GROUP_SIZE)));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s2 = fma(a.s2, b.s2, 0);
            sum0.s3 = fma(a.s3, b.s3, 0);

            sum0.s0 = fma(a.s4, b.s4, sum0.s0);
            sum0.s1 = fma(a.s5, b.s5, sum0.s1);
            sum0.s2 = fma(a.s6, b.s6, sum0.s2);
            sum0.s3 = fma(a.s7, b.s7, sum0.s3);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s2 = fma(a.s2, b2.s2, 0);
            sum1.s3 = fma(a.s3, b2.s3, 0);

            sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
            sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
            sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
            sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

            sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3];
            sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_gate_up)(
    const __global int* expert_list,
    const __global MOE_WEI_DT* gate_weight_addr,
    const __global MOE_SCALE_DT* gate_scale_addr,
    const __global MOE_ZP_DT* gate_zp_addr,
    const __global MOE_WEI_DT* up_weight_addr,
    const __global MOE_SCALE_DT* up_scale_addr,
    const __global MOE_ZP_DT* up_zp_addr,
#    if SHARED_EXPERT_ENABLE
    const __global MOE_WEI_DT* shared_gate_weight,
    const __global MOE_SCALE_DT* shared_gate_scale,
    const __global MOE_ZP_DT* shared_gate_zp,
    const __global MOE_WEI_DT* shared_up_weight,
    const __global MOE_SCALE_DT* shared_up_scale,
    const __global MOE_ZP_DT* shared_up_zp,
    const __global half* shared_gate_gate_weight,  // [HIDDEN_SIZE] (assuming no scale/zp for now, or pre-dequantized)
    __global MOE_DTYPE* shared_gate_out,           // [token_num] — output: sigmoid(dot(x, gate_weight)) for shared expert
#    endif
    __global MOE_DTYPE* x,    // [token_num, HIDDEN_SIZE]
    __global MOE_DTYPE* y) {  // [token_num * EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE]
    // global: [token_num*EXPERTS_PER_TOKEN, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    // Batched GEMV: each workgroup handles one (token, expert) pair.
    // For single token (token_num=1), flat_id == expert_slot and token_idx == 0.
    int flat_id = get_global_id(0);
    int token_idx = flat_id / EXPERTS_PER_TOKEN;
    int expert_slot = flat_id % EXPERTS_PER_TOKEN;
    y += flat_id * INTERMEDIATE_SIZE;

    // Check if we are processing the Shared Expert
#    if SHARED_EXPERT_ENABLE
    bool is_shared = (expert_slot == MAX_TOPK);
#    else
    bool is_shared = false;
#    endif

    // A3 per-GEMM strides (gate and up may differ in dtype/zp-mode).
    const int gate_expert_wei_size = MOE_EXPERT_WEI_BYTES(GATE_WEIGHT_DT);
    const int up_expert_wei_size = MOE_EXPERT_WEI_BYTES(UP_WEIGHT_DT);
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;  // f16 scale, dtype-independent
    const int gate_zp_stride = GATE_ZP_SCALAR ? 0 : MOE_EXPERT_ZP_BYTES(GATE_WEIGHT_DT, GATE_UP_GROUP_SIZE);
    const int up_zp_stride = UP_ZP_SCALAR ? 0 : MOE_EXPERT_ZP_BYTES(UP_WEIGHT_DT, GATE_UP_GROUP_SIZE);

    int expert_id = 0;
    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global MOE_WEI_DT* gate_weight;
    __global MOE_SCALE_DT* gate_scale;
    __global MOE_ZP_DT* gate_zp;
    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global MOE_WEI_DT* up_weight;
    __global MOE_SCALE_DT* up_scale;
    __global MOE_ZP_DT* up_zp;

    if (!is_shared) {
        expert_id = expert_list[token_idx * MAX_TOPK + expert_slot];
        // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
        gate_weight = (__global MOE_WEI_DT*)(gate_weight_addr + expert_id * gate_expert_wei_size);
        gate_scale = (__global MOE_SCALE_DT*)(gate_scale_addr + expert_id * expert_scale_size);
        gate_zp = (__global MOE_ZP_DT*)(gate_zp_addr + expert_id * gate_zp_stride);

        // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
        up_weight = (__global MOE_WEI_DT*)(up_weight_addr + expert_id * up_expert_wei_size);
        up_scale = (__global MOE_SCALE_DT*)(up_scale_addr + expert_id * expert_scale_size);
        up_zp = (__global MOE_ZP_DT*)(up_zp_addr + expert_id * up_zp_stride);
    }
#    if SHARED_EXPERT_ENABLE
    else {
        // Use shared expert pointers directly
        // [HIDDEN_SIZE, SHARED_INTERMEDIATE_SIZE] - assume layout/size match sparse experts for now
        gate_weight = (__global MOE_WEI_DT*)shared_gate_weight;
        gate_scale = (__global MOE_SCALE_DT*)shared_gate_scale;
        gate_zp = (__global MOE_ZP_DT*)shared_gate_zp;
        up_weight = (__global MOE_WEI_DT*)shared_up_weight;
        up_scale = (__global MOE_SCALE_DT*)shared_up_scale;
        up_zp = (__global MOE_ZP_DT*)shared_up_zp;
    }
#    endif

#    if GATE_UP_GROUP_SIZE % FAKE_GROUP_SIZE != 0
    if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
        printf("GATE_UP_GROUP_SIZE(%d) must be a multiple of FAKE_GROUP_SIZE(%d)", GATE_UP_GROUP_SIZE, FAKE_GROUP_SIZE);
    }
    return;
#    endif

    __local half x2[HIDDEN_SIZE];
#if HAS_ZP
    __local float xg_sum[HIDDEN_SIZE / FAKE_GROUP_SIZE];
#else
    __local float xg_sum[1];  // unused placeholder for function signature
#endif
#    if SHARED_EXPERT_ENABLE
    __local float shared_gate_partial[SUBGROUP_NUM];  // one slot per subgroup for scalar gate reduction
#    endif

#    if GATE_WEIGHT_DT == 0
    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + token_idx * HIDDEN_SIZE + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
#if HAS_ZP
            x_group_sum += even + odd;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    else
    //# load x into slm
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + token_idx * HIDDEN_SIZE + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
            half value = px[j];
            px2[j] = value;
#if HAS_ZP
            x_group_sum += value;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    endif

    barrier(CLK_LOCAL_MEM_FENCE);

#    if SHARED_EXPERT_ENABLE
    // Compute scalar gate for shared expert using all threads in the workgroup.
    // Only the N-block-0 workgroup writes shared_gate_out[token_idx]; other N-block workgroups
    // skip this entirely to avoid redundant work and races.
    if (is_shared && get_group_id(2) == 0) {
        // Step 1: every thread accumulates a partial dot product over its slice of HIDDEN_SIZE.
        // With num_sg subgroups * SUBGROUP_SIZE threads, each thread handles ~14 elements
        // (HIDDEN_SIZE=3584 / 256 threads) instead of one thread doing all 3584.
        float gate_val = 0.0f;
        int thread_id = id_sg * SUBGROUP_SIZE + id_local;
        int total_threads = num_sg * SUBGROUP_SIZE;
        for (int i = thread_id; i < HIDDEN_SIZE; i += total_threads) {
            gate_val += (float)x[token_idx * HIDDEN_SIZE + i] * (float)shared_gate_gate_weight[i];
        }

        // Step 2: intra-subgroup reduction.
        gate_val = sub_group_reduce_add(gate_val);

        // Step 3: store per-subgroup partial to SLM.
        if (id_local == 0) {
            shared_gate_partial[id_sg] = gate_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Step 4: subgroup 0 does final cross-subgroup reduction and writes sigmoid result.
        if (id_sg == 0) {
            float final_val = (id_local < num_sg) ? shared_gate_partial[id_local] : 0.0f;
            final_val = sub_group_reduce_add(final_val);
            if (id_local == 0) {
                shared_gate_out[token_idx] = (MOE_DTYPE)(1.0f / (1.0f + exp(-final_val)));
            }
        }
    }
    // shared_gate_out[token_idx] is consumed by the next kernel (mlp_down) — no barrier needed here.
#    endif

    // A3: dispatch up (silu=false) then gate (silu=true) by each projection's own dtype.
#    if UP_WEIGHT_DT == 0
    gate_up_gemv_n2x_u4(up_weight, up_scale, UP_SCALE_GLOBAL, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
#    elif UP_WEIGHT_DT == 1
    gate_up_gemv_n2x_u8(up_weight, up_scale, UP_SCALE_GLOBAL, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
#    elif UP_WEIGHT_DT == 2
    gate_up_gemv_n2x_f16(up_weight, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, false);
#    elif UP_WEIGHT_DT == 3
    gate_up_gemv_n2x_u2(up_weight, up_scale, UP_SCALE_GLOBAL, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false, UP_ZP_SCALAR);
#    endif
#    if GATE_WEIGHT_DT == 0
    gate_up_gemv_n2x_u4(gate_weight, gate_scale, GATE_SCALE_GLOBAL, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
#    elif GATE_WEIGHT_DT == 1
    gate_up_gemv_n2x_u8(gate_weight, gate_scale, GATE_SCALE_GLOBAL, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
#    elif GATE_WEIGHT_DT == 2
    gate_up_gemv_n2x_f16(gate_weight, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, true);
#    elif GATE_WEIGHT_DT == 3
    gate_up_gemv_n2x_u2(gate_weight, gate_scale, GATE_SCALE_GLOBAL, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true, GATE_ZP_SCALAR);
#    endif
}

#elif DOWN_ENABLE

inline void down_gemv_n2x_u4(const __global uchar* weight,
                             __global MOE_SCALE_DT* scales,
                             half scale_global,
                             __global uchar* zps,
                             MOE_DTYPE routing_weight_val,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
        __global MOE_SCALE_DT* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n / 2;
#endif
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half s0 = MOE_SCALE_AT(S, scale_offset, scale_global);
            half s1 = MOE_SCALE_AT(S, scale_offset + 1, scale_global);
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N / 2;
            ushort z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);
#endif

#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (1 byte = 2 nibbles).
            half sum0;
            half sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0 = fma(a.s0, (DEQUANT_4BIT_LO(b)), (half)0);
            sum0 = fma(a.s1, (DEQUANT_4BIT_HI(b)), sum0);

            sum1 = fma(a.s0, (DEQUANT_4BIT_LO(b2)), (half)0);
            sum1 = fma(a.s1, (DEQUANT_4BIT_HI(b2)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b.s2)), 0);
            sum0.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b.s3)), 0);

            sum0.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);
            sum0.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b.s2)), sum0.s2);
            sum0.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b.s3)), sum0.s3);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b2.s2)), 0);
            sum1.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b2.s3)), 0);

            sum1.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);
            sum1.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b2.s2)), sum1.s2);
            sum1.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b2.s3)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weight_val;
            y[n + 1] = sum_all1 * routing_weight_val;
        }
    }
}

inline void down_gemv_n2x_u2(const __global uchar* weight,
                             __global MOE_SCALE_DT* scales,
                             half scale_global,
                             __global uchar* zps,
                             MOE_DTYPE routing_weight_val,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 4;
        __global MOE_SCALE_DT* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n / 4;
        // n is even, so channels n and n+1 always share one packed zp byte.
        const int zshift = (n & 3) * 2;
#endif
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half s0 = MOE_SCALE_AT(S, scale_offset, scale_global);
            half s1 = MOE_SCALE_AT(S, scale_offset + 1, scale_global);
#if HAS_ZP
#        if DOWN_ZP_SCALAR
            const half z_hf0 = convert_half(((__global MOE_ZP_SCALAR_DT*)zps)[0]);
            const half z_hf1 = z_hf0;
#        else
            const uchar z = Z[(gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N / 4];
            const half z_hf0 = convert_half((z >> zshift) & 0x3);
            const half z_hf1 = convert_half((z >> (zshift + 2)) & 0x3);
#        endif
#endif

#    if ELEMS_PER_LANE == 4
            // Each lane reads 4 K-elements (1 byte = 4 x 2-bit values).
            half2 sum0;
            half2 sum1;
            half4 a = vload4(id_local, x2 + gk * FAKE_GROUP_SIZE);
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 4);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 4) + gk * FAKE_GROUP_SIZE / 4));

            sum0.s0 = fma(a.s0, (DEQUANT_2BIT(b, 0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_2BIT(b, 2)), 0);
            sum0.s0 = fma(a.s2, (DEQUANT_2BIT(b, 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (DEQUANT_2BIT(b, 6)), sum0.s1);

            sum1.s0 = fma(a.s0, (DEQUANT_2BIT(b2, 0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_2BIT(b2, 2)), 0);
            sum1.s0 = fma(a.s2, (DEQUANT_2BIT(b2, 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (DEQUANT_2BIT(b2, 6)), sum1.s1);

#    if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#    endif
#    elif ELEMS_PER_LANE == 2
            // Each lane handles 2 K-elements; two lanes share one packed byte.
            // NOTE: keep per-lane scalar loads here. A subgroup block read would make
            // lanes >= FAKE_GROUP_SIZE/4 read past the tile, and at the end of the
            // mmap'd weights buffer that overhang faults (CL_OUT_OF_RESOURCES).
            half sum0;
            half sum1;
            half2 a = vload2(id_local, x2 + gk * FAKE_GROUP_SIZE);
            uchar b = B[gk * FAKE_GROUP_SIZE / 4 + (id_local >> 1)];
            uchar b2 = B[(K / 4) + gk * FAKE_GROUP_SIZE / 4 + (id_local >> 1)];
            const int wshift = (id_local & 1) * 4;

            sum0 = fma(a.s0, (DEQUANT_2BIT(b, wshift)), (half)0);
            sum0 = fma(a.s1, (DEQUANT_2BIT(b, wshift + 2)), sum0);

            sum1 = fma(a.s0, (DEQUANT_2BIT(b2, wshift)), (half)0);
            sum1 = fma(a.s1, (DEQUANT_2BIT(b2, wshift + 2)), sum1);

#    if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#    endif
#    elif ELEMS_PER_LANE == 1
            // Each lane handles 1 K-element; four lanes share one packed byte.
            // Same per-lane scalar loads as the ==2 branch: subgroup block reads
            // would overhang the weights buffer and fault (CL_OUT_OF_RESOURCES).
            half sum0;
            half sum1;
            half a = x2[gk * FAKE_GROUP_SIZE + id_local];
            uchar b = B[gk * FAKE_GROUP_SIZE / 4 + (id_local >> 2)];
            uchar b2 = B[(K / 4) + gk * FAKE_GROUP_SIZE / 4 + (id_local >> 2)];
            const int wshift = (id_local & 3) * 2;

            sum0 = fma(a, (DEQUANT_2BIT(b, wshift)), (half)0);
            sum1 = fma(a, (DEQUANT_2BIT(b2, wshift)), (half)0);

#    if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#    endif
#    else
            half4 sum0;
            half4 sum1;
            // The block read is strided: lane L gets byte L and byte L+SUBGROUP_SIZE, which cover
            // K elements 4L.. and 4L+FAKE_GROUP_SIZE/2.. respectively. The activations must be
            // loaded in two matching contiguous quads, not as one vload8 of 8L...
            half4 alo = vload4(id_local, x2 + gk * FAKE_GROUP_SIZE);
            half4 ahi = vload4(id_local, x2 + gk * FAKE_GROUP_SIZE + FAKE_GROUP_SIZE / 2);
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 4);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 4) + gk * FAKE_GROUP_SIZE / 4));

            sum0.s0 = fma(alo.s0, (DEQUANT_2BIT(b.s0, 0)), 0);
            sum0.s1 = fma(alo.s1, (DEQUANT_2BIT(b.s0, 2)), 0);
            sum0.s2 = fma(alo.s2, (DEQUANT_2BIT(b.s0, 4)), 0);
            sum0.s3 = fma(alo.s3, (DEQUANT_2BIT(b.s0, 6)), 0);

            sum0.s0 = fma(ahi.s0, (DEQUANT_2BIT(b.s1, 0)), sum0.s0);
            sum0.s1 = fma(ahi.s1, (DEQUANT_2BIT(b.s1, 2)), sum0.s1);
            sum0.s2 = fma(ahi.s2, (DEQUANT_2BIT(b.s1, 4)), sum0.s2);
            sum0.s3 = fma(ahi.s3, (DEQUANT_2BIT(b.s1, 6)), sum0.s3);

            sum1.s0 = fma(alo.s0, (DEQUANT_2BIT(b2.s0, 0)), 0);
            sum1.s1 = fma(alo.s1, (DEQUANT_2BIT(b2.s0, 2)), 0);
            sum1.s2 = fma(alo.s2, (DEQUANT_2BIT(b2.s0, 4)), 0);
            sum1.s3 = fma(alo.s3, (DEQUANT_2BIT(b2.s0, 6)), 0);

            sum1.s0 = fma(ahi.s0, (DEQUANT_2BIT(b2.s1, 0)), sum1.s0);
            sum1.s1 = fma(ahi.s1, (DEQUANT_2BIT(b2.s1, 2)), sum1.s1);
            sum1.s2 = fma(ahi.s2, (DEQUANT_2BIT(b2.s1, 4)), sum1.s2);
            sum1.s3 = fma(ahi.s3, (DEQUANT_2BIT(b2.s1, 6)), sum1.s3);

#    if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#    endif
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weight_val;
            y[n + 1] = sum_all1 * routing_weight_val;
        }
    }
}

inline void down_gemv_n2x_u8(const __global uchar* weight,
                             __global MOE_SCALE_DT* scales,
                             half scale_global,
                             __global uchar* zps,
                             MOE_DTYPE routing_weight_val,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K;
        __global MOE_SCALE_DT* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n;
#endif
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half s0 = MOE_SCALE_AT(S, scale_offset, scale_global);
            half s1 = MOE_SCALE_AT(S, scale_offset + 1, scale_global);
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half z0 = convert_half(Z[zp_offset]);
            half z1 = convert_half(Z[zp_offset + 1]);
#endif

#    if ELEMS_PER_LANE == 4
            float2 sum0;
            float2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)B + K + gk * FAKE_GROUP_SIZE);

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), sum0.s0);
            sum0.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), sum0.s1);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), sum1.s0);
            sum1.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (8-bit weights, 1 byte each).
            float sum0;
            float sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)B + K + gk * FAKE_GROUP_SIZE);

            sum0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), sum0);

            sum1 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            float4 sum0;
            float4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), 0.0f);
            sum0.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), 0.0f);

            sum0.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b.s4)), sum0.s0);
            sum0.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b.s5)), sum0.s1);
            sum0.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b.s6)), sum0.s2);
            sum0.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b.s7)), sum0.s3);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), 0.0f);
            sum1.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), 0.0f);

            sum1.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b2.s4)), sum1.s0);
            sum1.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b2.s5)), sum1.s1);
            sum1.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b2.s6)), sum1.s2);
            sum1.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b2.s7)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weight_val;
            y[n + 1] = sum_all1 * routing_weight_val;
        }
    }
}

inline void down_gemv_n2x_f16(const __global half* weight, MOE_DTYPE routing_weight_val, __global half* y, int N, int K, half* x2) {
    int id_local = get_sub_group_local_id();
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global half* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {

#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s0 = fma(a.s2, b.s2, sum0.s0);
            sum0.s1 = fma(a.s3, b.s3, sum0.s1);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
            sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

            sum_all0 += sum0[0] + sum0[1];
            sum_all1 += sum1[0] + sum1[1];
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 fp16 elements.
            half sum0;
            half sum1;
            half2 a = as_half2(intel_sub_group_block_read_us2((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half2 b = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half2 b2 = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0 = fma(a.s0, b.s0, (half)0);
            sum0 = fma(a.s1, b.s1, sum0);

            sum1 = fma(a.s0, b2.s0, (half)0);
            sum1 = fma(a.s1, b2.s1, sum1);

            sum_all0 += sum0;
            sum_all1 += sum1;
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half8 b2 = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s2 = fma(a.s2, b.s2, 0);
            sum0.s3 = fma(a.s3, b.s3, 0);

            sum0.s0 = fma(a.s4, b.s4, sum0.s0);
            sum0.s1 = fma(a.s5, b.s5, sum0.s1);
            sum0.s2 = fma(a.s6, b.s6, sum0.s2);
            sum0.s3 = fma(a.s7, b.s7, sum0.s3);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s2 = fma(a.s2, b2.s2, 0);
            sum1.s3 = fma(a.s3, b2.s3, 0);

            sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
            sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
            sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
            sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

            sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3];
            sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weight_val;
            y[n + 1] = sum_all1 * routing_weight_val;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_down)(const __global int* expert_list,
                                                                           const __global MOE_WEI_DT* down_weight_addr,
                                                                           const __global MOE_SCALE_DT* down_scale_addr,
                                                                           const __global MOE_ZP_DT* down_zp_addr,
#    if SHARED_EXPERT_ENABLE
                                                                           const __global MOE_WEI_DT* shared_down_weight,
                                                                           const __global MOE_SCALE_DT* shared_down_scale,
                                                                           const __global MOE_ZP_DT* shared_down_zp,
#    endif
                                                                           const __global MOE_DTYPE* x,          // [token_num * EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE]
                                                                           const __global MOE_DTYPE* routing_weights,  // [token_num * MAX_TOPK] compact buffer from MoERouterFused
#    if SHARED_EXPERT_ENABLE
                                                                           const __global MOE_DTYPE* shared_gate_in,   // [token_num] separate shared expert gate values
#    endif
                                                                           __global MOE_DTYPE* y) {              // [token_num * EXPERTS_PER_TOKEN, HIDDEN_SIZE]
    // global: [token_num*EXPERTS_PER_TOKEN, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int flat_id = get_global_id(0);
    int token_idx = flat_id / EXPERTS_PER_TOKEN;
    int expert_slot = flat_id % EXPERTS_PER_TOKEN;
    x += flat_id * INTERMEDIATE_SIZE;
    y += flat_id * HIDDEN_SIZE;

#    if SHARED_EXPERT_ENABLE
    bool is_shared = (expert_slot == MAX_TOPK);
#    else
    bool is_shared = false;
#    endif

    // A3 per-GEMM strides for the down projection.
    const int expert_wei_size = MOE_EXPERT_WEI_BYTES(DOWN_WEIGHT_DT);
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / DOWN_GROUP_SIZE;  // f16 scale, dtype-independent
    const int down_zp_stride = DOWN_ZP_SCALAR ? 0 : MOE_EXPERT_ZP_BYTES(DOWN_WEIGHT_DT, DOWN_GROUP_SIZE);
    int expert_id = 0;

    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global MOE_WEI_DT* weight;
    __global MOE_SCALE_DT* scales;
    __global MOE_ZP_DT* zps;

    if (!is_shared) {
        expert_id = expert_list[token_idx * MAX_TOPK + expert_slot];
        // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
        weight = (__global MOE_WEI_DT*)(down_weight_addr + expert_id * expert_wei_size);
        scales = (__global MOE_SCALE_DT*)(down_scale_addr + expert_id * expert_scale_size);
        zps = (__global MOE_ZP_DT*)(down_zp_addr + expert_id * down_zp_stride);
    }
#    if SHARED_EXPERT_ENABLE
    else {
        weight = (__global MOE_WEI_DT*)shared_down_weight;
        scales = (__global MOE_SCALE_DT*)shared_down_scale;
        zps = (__global MOE_ZP_DT*)shared_down_zp;
    }
#    endif

#    if DOWN_GROUP_SIZE % FAKE_GROUP_SIZE != 0
    if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
        printf("DOWN_GROUP_SIZE(%d) must be a multiple of FAKE_GROUP_SIZE(%d)", DOWN_GROUP_SIZE, FAKE_GROUP_SIZE);
    }
    return;
#    endif

    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;

    __local half x2[INTERMEDIATE_SIZE];
#if HAS_ZP
    __local float xg_sum[INTERMEDIATE_SIZE / FAKE_GROUP_SIZE];
#else
    __local float xg_sum[1];  // unused placeholder for function signature
#endif

#    if DOWN_WEIGHT_DT == 0
    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < INTERMEDIATE_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
#if HAS_ZP
            x_group_sum += even + odd;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    else
    //# load x into slm
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < INTERMEDIATE_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
            half value = px[j];
            px2[j] = value;
#if HAS_ZP
            x_group_sum += value;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    endif

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute the routing weight for this (token, expert) work item.
    // routing_weights is compact [token_num * MAX_TOPK]; shared expert gate is separate.
#    if SHARED_EXPERT_ENABLE
    MOE_DTYPE routing_weight_val = is_shared ? shared_gate_in[token_idx] : routing_weights[token_idx * MAX_TOPK + expert_slot];
#    else
    MOE_DTYPE routing_weight_val = routing_weights[token_idx * MAX_TOPK + expert_slot];
#    endif

#    if DOWN_WEIGHT_DT == 0
    down_gemv_n2x_u4(weight, scales, DOWN_SCALE_GLOBAL, zps, routing_weight_val, y, N, K, x2, xg_sum);
#    elif DOWN_WEIGHT_DT == 1
    down_gemv_n2x_u8(weight, scales, DOWN_SCALE_GLOBAL, zps, routing_weight_val, y, N, K, x2, xg_sum);
#    elif DOWN_WEIGHT_DT == 2
    down_gemv_n2x_f16(weight, routing_weight_val, y, N, K, x2);
#    elif DOWN_WEIGHT_DT == 3
    down_gemv_n2x_u2(weight, scales, DOWN_SCALE_GLOBAL, zps, routing_weight_val, y, N, K, x2, xg_sum);
#    endif
}

// mlp_reduce is the fallback group: it is selected when none of the specific *_ENABLE flags are
// set. The condition must exclude U2_GEMM_ENABLE explicitly rather than being a bare #else,
// because a group after #else cannot be an #elif (C99/OpenCL C 6.10.1) and because mlp_reduce
// needs MOE_DTYPE / HIDDEN_SIZE / MAX_TOPK, none of which are defined for the u2 GEMM build.
#elif !defined(U2_GEMM_ENABLE)
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_reduce)(const __global MOE_DTYPE* x,  // [token_num * REDUCE_COUNT, HIDDEN_SIZE]
                                                                             __global MOE_DTYPE* y) {      // [token_num, HIDDEN_SIZE]
    // gws={token_num, HIDDEN_SIZE}, lws={1, min(max_wgs, 1024)}
    int token_idx = get_global_id(0);
    int n = get_global_id(1);
#    if SHARED_EXPERT_ENABLE
#        define REDUCE_COUNT (MAX_TOPK + 1)
#    else
#        define REDUCE_COUNT MAX_TOPK
#    endif
    half sum[REDUCE_COUNT] = {0};
    __attribute__((opencl_unroll_hint(REDUCE_COUNT))) for (int i = 0; i < REDUCE_COUNT; i++) {
        sum[i] = as_half(intel_sub_group_block_read_us((const __global ushort*)(x + (token_idx * REDUCE_COUNT + i) * HIDDEN_SIZE + n)));
    }
    for (int i = 1; i < REDUCE_COUNT; i++) {
        sum[0] += sum[i];
    }
    intel_sub_group_block_write_us((__global ushort*)(y + token_idx * HIDDEN_SIZE + n), as_ushort(sum[0]));
}

#elif defined(U2_GEMM_ENABLE)

// Native u2 grouped GEMM for prefill. Replaces "unpack u2 -> u4 scratch, then run the u4
// oneDNN grouped matmul", which cost a permanent 12.08 GB of device memory on
// Qwen3.6-35B-A3B (2x the 5.63 GiB of u2 expert weights) plus a ~2.0 s one-time unpack.
//
// Why a hand-written kernel rather than the micro-gemm path: micro-gemm hands the weight
// dtype to oneDNN's JIT GEMM generator (gemmstone), whose Type has no 2-bit entry and whose
// address arithmetic hard-codes "at most 2 elements per byte" (is4()/perByte()/log2PerByte(),
// with subByteCheck() stubbing anything narrower). Adding u2 there means modifying vendored
// oneDNN. Raising batched_gemv_threshold does not work either: mlp_gate_up is one work-group
// per (token, expert) pair, so M tokens routed to one expert re-read that expert's weights M
// times — the GEMV amortizes nothing along M.
//
// This kernel is deliberately a minimal extension of the *verified* gate_up_gemv_n2x_u2: the
// weight/scale/zp addressing and the dequant sequence are reproduced verbatim, and the only
// structural change is that SLM holds TILE_M tokens instead of one. A single
// intel_sub_group_block_read_uc of weights then feeds TILE_M FMAs, and the shift/mask dequant
// is likewise paid once per TILE_M instead of once per token.
//
// Addressing note: the scale expression below (S[n + group * N]) is copied from the GEMV
// rather than derived from the IR const shape, which reads as [E, N, G] and would imply
// n * G + group. The two disagree, but the GEMV demonstrably produces correct output, so
// something reorders scales between the IR constant and this kernel's argument. Copying the
// verified expression is the only safe choice until a numeric GPU-vs-CPU probe settles it.
//
// Dispatch (work-groups never straddle an expert boundary):
//   global = [num_blocks, SUBGROUP_SIZE, U2_GEMM_N / N_BLOCK]
//   local  = [1, SUBGROUP_SIZE, SUBGROUP_NUM]
//   blocks[b] = {expert_id, token_start, n_tokens}, n_tokens <= TILE_M, built on the host from
//   the per-expert token counts that exec_prefill_grouped_gemm already computes.

#    if defined(U2_DPAS_ENABLE)

// ---------------------------------------------------------------------------------------------
// DPAS (systolic) variant.
//
// The FMA variant below is correct but spends 20% of the machine: it stages TILE_M*K halfs of
// activations in SLM (33792 B at K=2048), which is past the 32 KB SLM granule, so only 2
// work-groups fit per Xe core = 16 of 80 hardware threads. Three controlled measurements pin the
// blame on SLM residency rather than on the algorithm: the down GEMM (K=512, 8448 B, 10 WG/core)
// runs at 992 GMAC/s against gate/up's 555 GMAC/s on the same instruction stream; dropping TILE_M
// 8 -> 7 (29568 B, back under the granule) takes the 1k gate/up shape from 16.4 to 9.3 ms.
//
// So this variant does not tile the SLM staging - it deletes it. Operand roles are rotated so that
// nothing but the zero-point row sums needs to be shared:
//   lane            = output channel  (was: a slice of K, folded by sub_group_reduce_add)
//   acc component   = token row
//   weights -> `b`  = one uint of the [E, N, K] u2 buffer per lane per k16 step. 16 consecutive K
//                     of one channel is exactly 4 bytes of the existing layout, so the weights
//                     feed the systolic array with no transpose, no SLM and no repacking.
//   activations -> `a` = read straight from global with a block2d load.
// SLM drops 33792 B -> 4096 B (gate/up) / 1024 B (down), occupancy 20% -> 100%, and the
// sub_group_reduce_add and the single-lane scalar stores both disappear.
//
// Measured on Arc B390 against the FMA variant, same session:
//   gate/up   55 tok  2.723 ms -> 0.535 ms   gate/up 1k  16.426 ms -> 1.169 ms
//   down      55 tok  1.635 ms -> 0.521 ms   down    1k   9.485 ms -> 1.159 ms
// At those times the kernel moves 42.5-96 MB of weights at 81-103 GB/s against a measured 113
// GB/s streaming ceiling, i.e. it is at the DRAM floor, not ALU-bound. Deleting the whole xg_sum
// pass changes nothing measurable. That is why no further dequant cleverness (k-permuted unpack,
// int8 DPAS) is implemented here: there is nothing left to buy.
//
// Do not "optimise" the M tile upward. 48 and 64 rows measure 1.63 and 2.08 ms against 1.17 ms at
// 32 rows for the same 1k shape, despite halving weight re-reads.

#        define U2_MTILE    ((U2_MSUB) * 8)
#        define U2_NUM_G    ((U2_GEMM_K) / (U2_GEMM_GROUP_SIZE))
#        define U2_N_PER_WG ((U2_N_SG) * 16)

// The block2d activation load needs a 64-byte-aligned base and a pitch that is a multiple of 16
// bytes; K % 32 == 0 gives a pitch of at least 64 bytes and keeps a quant group from straddling a
// 16-deep DPAS step. The host keeps the FMA variant as the fallback when these do not hold.
#        if (U2_GEMM_K) % 32 != 0
#            error "u2 DPAS GEMM needs U2_GEMM_K % 32 == 0"
#        endif
#        if (U2_GEMM_GROUP_SIZE) % 32 != 0
#            error "u2 DPAS GEMM needs U2_GEMM_GROUP_SIZE % 32 == 0"
#        endif
#        if (U2_GEMM_N) % (U2_N_PER_WG) != 0
#            error "u2 DPAS GEMM needs U2_GEMM_N % (U2_N_SG * 16) == 0"
#        endif
// The zero point is either a folded per-tensor scalar (INT2_SYM) or per (group, channel) u2
// (INT2_ASYM), and which one is a per-GEMM property: a mixed layer can pair a scalar gate with a
// per-group down. This kernel is instantiated once per GEMM, so the host answers it per instance.
#        if HAS_ZP && !defined(U2_GEMM_ZP_SCALAR)
#            error "u2 DPAS GEMM: the host must define U2_GEMM_ZP_SCALAR (0/1) whenever HAS_ZP"
#        endif

// SUBGROUP_SIZE is 32 on xe2+ and intel_sub_group_f16_f16_matrix_mad_k16 *compiles* at 32 while
// producing wrong results under every lane mapping. Hence the separate U2_DPAS_SG: this kernel
// must never reference SUBGROUP_SIZE.
__attribute__((intel_reqd_sub_group_size(U2_DPAS_SG))) KERNEL(moe_u2_gemm)(
    const __global half* a_ptr,       // [total_gathered, U2_GEMM_K], sorted by expert
    const __global uchar* wei_ptr,    // [E, U2_GEMM_N, U2_GEMM_K/4] u2, K innermost
    const __global MOE_SCALE_DT* scale_ptr,  // per-group scales (f16, or e4m3 + U2_GEMM_SCALE_GLOBAL)
    const __global uchar* zp_ptr,     // scalar zp (INT2_SYM) or per-group u2 [E, G, N] (INT2_ASYM)
    const __global int* blocks,       // [num_blocks * 3] {expert_id, token_start, n_tokens}
    // Output last: execute_stage() binds every INPUT descriptor before any OUTPUT one.
    __global half* c_ptr) {           // [total_gathered, U2_GEMM_N]
    const int lane = get_sub_group_local_id();
    const int sgid = get_local_id(2);
    const int block_idx = get_group_id(2);
    const int expert_id = blocks[block_idx * 3 + 0];
    const int token_start = blocks[block_idx * 3 + 1];
    const int n_tokens = blocks[block_idx * 3 + 2];

    const int N = U2_GEMM_N;
    const int K = U2_GEMM_K;

    // get_group_id(0) is the N tile and varies fastest, so the work-groups that share an
    // activation tile stay co-resident and hit it in L2.
    const int n0 = (int)get_group_id(0) * U2_N_PER_WG + sgid * 16;

    const __global uchar* weight = wei_ptr + (size_t)expert_id * ((size_t)N * K / 4);
    const __global MOE_SCALE_DT* scales = scale_ptr + (size_t)expert_id * ((size_t)N * K / U2_GEMM_GROUP_SIZE);

    // The only SLM left. Per token, per quant group, the plain sum of activations, for the
    // zero-point correction. NOTE this is the FULL sum: unlike the FMA variant there is no
    // sub_group_reduce_add afterwards, so it must NOT be pre-divided by the subgroup size.
    __local float xg[U2_MTILE * U2_NUM_G];
    {
        const int lid = sgid * (U2_DPAS_SG) + lane;
        const int nthreads = (U2_N_SG) * 16;
        for (int idx = lid; idx < U2_MTILE * U2_NUM_G; idx += nthreads) {
            const int m = idx / U2_NUM_G;
            const int g = idx % U2_NUM_G;
            float s = 0.0f;
            if (m < n_tokens) {
                const __global half* r = a_ptr + (size_t)(token_start + m) * K + g * (U2_GEMM_GROUP_SIZE);
                for (int e = 0; e < (U2_GEMM_GROUP_SIZE); e++) {
                    s += convert_float(r[e]);
                }
            }
            xg[idx] = s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // This lane's output channel, K innermost, so 4 bytes = 16 consecutive K.
    const __global uint* wrow = (const __global uint*)(weight + (size_t)(n0 + lane) * (K / 4));

    // Nothing is dequantised into f16. The weight enters the systolic array as the biased integer
    // 1024 + 256*w (see the magic-bias unpack below) and the entire correction lands on the f32
    // accumulator instead:
    //   out = sum_g (s_g / 256) * ( D_g - 256*(4 + zp) * Sa_g )
    // where D_g is the raw DPAS accumulation over group g and Sa_g the undivided xg sum. The
    // 1024 offset contributes even when there is no zero point, so unlike the FMA variant xg is
    // computed unconditionally.
#        if !HAS_ZP
    const float zeff = 256.0f * 4.0f;
#        elif U2_GEMM_ZP_SCALAR
    // Folded per-tensor zp: the whole correction factor is loop-invariant.
    const float zeff = 256.0f * (4.0f + convert_float(((const __global MOE_ZP_SCALAR_DT*)zp_ptr)[0]));
#        else
    // Per (group, channel) zp. Layout is [E, G, N] with N innermost -- byfx, which
    // prepare_quantization.cpp applies to the zp exactly as to the scales -- and u2-packed 4 per
    // byte, LSB-first: the same layout and packing gate_up_gemv_n2x_u2 indexes. lane == output
    // channel, so this sub-group's 16 channels live in 4 consecutive bytes and each lane extracts
    // its own 2-bit field. N % 4 == 0 (N is a multiple of U2_N_SG * 16) is what makes it legal to
    // add the group-row offset and the channel offset separately. zeff itself is per group and is
    // therefore computed inside the group loop below.
    const __global uchar* zps = zp_ptr + (size_t)expert_id * ((size_t)N * (U2_NUM_G) / 4);
    const int zshift = ((n0 + lane) & 3) * 2;
#        endif

    // Clamp the block2d surface to this block's own last row. Rows past `height` read back as
    // hardware zeros, which self-pads the ragged final token block (no zero-fill loop) and makes
    // it impossible for a work-group to touch another expert's tokens.
    const int a_height = token_start + n_tokens;

    float8 acc[U2_MSUB];
    unroll_for(int i = 0; i < U2_MSUB; i++) {
        acc[i] = (float8)(0.0f);
    }

    for (int g = 0; g < U2_NUM_G; g++) {
        // lane == channel, so this group's 16 scales are contiguous: one block read.
        // Indexing is S[group * N + n], copied from gate_up_gemv_n2x_u2 rather than derived from
        // the IR const shape - see the note on the FMA variant below.
        const float sf = MOE_SCALE_BLOCK_READ(scales + (size_t)g * N + n0, U2_GEMM_SCALE_GLOBAL);

#        if HAS_ZP && !U2_GEMM_ZP_SCALAR
        // One byte load and two ALU ops per quant group, against U2_GEMM_GROUP_SIZE/16 DPAS ops.
        // The correction already sits in per-lane registers (g8), so a per-channel zeff is free.
        // Byte index (g*N + n0 + lane)/4, split into g*N/4 + (n0+lane)/4 because N % 4 == 0.
        const float zeff =
            256.0f * (4.0f + convert_float((zps[(size_t)g * N / 4 + ((n0 + lane) >> 2)] >> zshift) & 0x3));
#        endif
        float8 g8[U2_MSUB];
        unroll_for(int i = 0; i < U2_MSUB; i++) {
            if (i * 8 < n_tokens) {
                unroll_for(int m = 0; m < 8; m++) {
                    g8[i][m] = -zeff * xg[(i * 8 + m) * U2_NUM_G + g];
                }
            }
        }

        // One weight gather per group, hoisted above the M loop so the unpack is paid once per
        // U2_MTILE token rows. 16-byte aligned because K/4 is a multiple of 16 here.
#        if (U2_GEMM_GROUP_SIZE) == 32
        const uint2 pk = vload2(0, wrow + g * ((U2_GEMM_GROUP_SIZE) / 16));
#        elif (U2_GEMM_GROUP_SIZE) == 64
        const uint4 pk = vload4(0, wrow + g * ((U2_GEMM_GROUP_SIZE) / 16));
#        elif (U2_GEMM_GROUP_SIZE) == 128
        const uint8 pk = vload8(0, wrow + g * ((U2_GEMM_GROUP_SIZE) / 16));
#        else
#            error "u2 DPAS GEMM supports group_size 32, 64 or 128"
#        endif

        unroll_for(int kb = 0; kb < (U2_GEMM_GROUP_SIZE) / 32; kb++) {
            int8 b0, b1;
            const uint p0 = pk[2 * kb];
            const uint p1 = pk[2 * kb + 1];
            // Magic bias: 0x6400 is the f16 1024, whose mantissa bits 8-9 are worth 256 each, so
            // OR-ing the 2-bit weight into them yields exactly f16(1024 + 256*w) in 5 ops per
            // k-pair. b[j] packs k=k0+2j in the low half and k=k0+2j+1 in the high half (VNNI).
            unroll_for(int j = 0; j < 8; j++) {
                const uint u = p0 >> (4 * j);
                b0[j] = (int)(0x64006400u | ((u << 8) & 0x00000300u) | ((u << 22) & 0x03000000u));
            }
            unroll_for(int j = 0; j < 8; j++) {
                const uint u = p1 >> (4 * j);
                b1[j] = (int)(0x64006400u | ((u << 8) & 0x00000300u) | ((u << 22) & 0x03000000u));
            }
            const int k0 = g * (U2_GEMM_GROUP_SIZE) + kb * 32;
            unroll_for(int i = 0; i < U2_MSUB; i++) {
                // n_tokens comes from the block list, so this is sub-group uniform and the mads
                // stay in uniform control flow.
                if (i * 8 >= n_tokens) {
                    continue;
                }
                const ushort16 araw = intel_subgroup_block_read_u16_m8k16v2((__global void*)a_ptr,
                                                                            K * 2,
                                                                            a_height,
                                                                            K * 2,
                                                                            (int2)(k0, token_start + i * 8));
                g8[i] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(araw.lo), b0, g8[i]);
                g8[i] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(araw.hi), b1, g8[i]);
            }
        }

        const float sc = sf * (1.0f / 256.0f);
        unroll_for(int i = 0; i < U2_MSUB; i++) {
            if (i * 8 < n_tokens) {
                unroll_for(int m = 0; m < 8; m++) {
                    acc[i][m] = fma(g8[i][m], sc, acc[i][m]);
                }
            }
        }
    }

    // lane == channel, so a whole 16-channel row goes out in one block write.
    unroll_for(int i = 0; i < U2_MSUB; i++) {
        unroll_for(int m = 0; m < 8; m++) {
            const int row = i * 8 + m;
            if (row < n_tokens) {
                intel_sub_group_block_write_us(
                    (__global ushort*)(c_ptr + (size_t)(token_start + row) * N + n0),
                    as_ushort(convert_half(acc[i][m])));
            }
        }
    }
}

#    else  // !U2_DPAS_ENABLE - portable FMA fallback, kept for shapes the block2d load rejects

#    if (U2_GEMM_K) % FAKE_GROUP_SIZE != 0
#        error "U2_GEMM_K must be a multiple of FAKE_GROUP_SIZE"
#    endif
#    if (U2_GEMM_GROUP_SIZE) % FAKE_GROUP_SIZE != 0
#        error "U2_GEMM_GROUP_SIZE must be a multiple of FAKE_GROUP_SIZE"
#    endif
// This kernel has no ELEMS_PER_LANE 8 branch (reachable at SUBGROUP_SIZE 16 with group_size >= 128).
// MoE3GemmSwigluU2Gemm::fma_supported() keeps the host from selecting this variant there, so this
// is the backstop for the two drifting apart: reject loudly instead of silently taking a
// wrong-width branch.
#    if ELEMS_PER_LANE != 1 && ELEMS_PER_LANE != 2 && ELEMS_PER_LANE != 4
#        error "u2 GEMM supports ELEMS_PER_LANE 1, 2 or 4 only"
#    endif
// Same per-GEMM zero-point form as the DPAS variant above.
#    if HAS_ZP && !defined(U2_GEMM_ZP_SCALAR)
#        error "u2 GEMM: the host must define U2_GEMM_ZP_SCALAR (0/1) whenever HAS_ZP"
#    endif

#    define U2_GEMM_NUM_GK ((U2_GEMM_K) / FAKE_GROUP_SIZE)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(moe_u2_gemm)(
    const __global half* a_ptr,       // [total_gathered, U2_GEMM_K], sorted by expert
    const __global uchar* wei_ptr,    // [E, U2_GEMM_N, U2_GEMM_K/4] u2, K innermost
    const __global MOE_SCALE_DT* scale_ptr,  // per-group scales (f16, or e4m3 + U2_GEMM_SCALE_GLOBAL)
    const __global uchar* zp_ptr,     // scalar zp (INT2_SYM) or per-group u2 [E, G, N] (INT2_ASYM)
    const __global int* blocks,       // [num_blocks * 3] {expert_id, token_start, n_tokens}
    // Output last: execute_stage() binds every INPUT descriptor before any OUTPUT one, so the
    // kernel signature has to list all read-only buffers first.
    __global half* c_ptr) {           // [total_gathered, U2_GEMM_N]
    const int id_local = get_sub_group_local_id();
    const int block_idx = get_group_id(0);
    const int expert_id = blocks[block_idx * 3 + 0];
    const int token_start = blocks[block_idx * 3 + 1];
    const int n_tokens = blocks[block_idx * 3 + 2];

    const int N = U2_GEMM_N;
    const int K = U2_GEMM_K;

    // Per-expert bases. Byte stride for u2 is N*K/4; the scale stride is dtype-independent.
    const __global uchar* weight = wei_ptr + (size_t)expert_id * ((size_t)N * K / 4);
    const __global MOE_SCALE_DT* scales = scale_ptr + (size_t)expert_id * ((size_t)N * K / U2_GEMM_GROUP_SIZE);
#    if HAS_ZP && !U2_GEMM_ZP_SCALAR
    // Per (group, channel) zp: N * (K/group_size) entries per expert, u2-packed 4 per byte, in the
    // same [E, G, N] byfx layout as the scales.
    const __global uchar* zps = zp_ptr + (size_t)expert_id * (((size_t)N * (K / U2_GEMM_GROUP_SIZE)) / 4);
#    endif

    // Stage TILE_M token rows in SLM. Every subgroup in this work-group sweeps the same rows
    // across its own N_BLOCK output channels, so the staging cost is shared SUBGROUP_NUM ways.
    __local half x2[TILE_M * U2_GEMM_K];
#    if HAS_ZP
    // Per-token, per-group sum of activations, for the (w - zp) * s correction. The GEMV keeps
    // one row of this; the GEMM needs one row per staged token.
    __local float xg_sum[TILE_M * U2_GEMM_NUM_GK];
#    else
    __local float xg_sum[1];  // unused placeholder
#    endif

    {
        const int lid = get_local_id(2) * SUBGROUP_SIZE + id_local;
        const int nthreads = get_local_size(2) * SUBGROUP_SIZE;
        for (int m = 0; m < n_tokens; m++) {
            const __global half* src = a_ptr + (size_t)(token_start + m) * K;
            for (int k = lid; k < K; k += nthreads) {
                x2[m * K + k] = src[k];
            }
        }
        // Zero the tail so a partial block (n_tokens < TILE_M) contributes nothing.
        for (int m = n_tokens; m < TILE_M; m++) {
            for (int k = lid; k < K; k += nthreads) {
                x2[m * K + k] = (half)0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#    if HAS_ZP
        for (int idx = lid; idx < TILE_M * U2_GEMM_NUM_GK; idx += nthreads) {
            const int m = idx / U2_GEMM_NUM_GK;
            const int gk = idx % U2_GEMM_NUM_GK;
            float acc = 0.0f;
            for (int e = 0; e < FAKE_GROUP_SIZE; e++) {
                acc += convert_float(x2[m * K + gk * FAKE_GROUP_SIZE + e]);
            }
            // Pre-divide by SUBGROUP_SIZE, exactly as the gold reference does at its
            // "xg_sum[i] = x_group_sum / SUBGROUP_SIZE" line. The zp correction term
            // (xg_sum * zp) is subtracted inside EVERY lane and only afterwards folded by
            // sub_group_reduce_add, so it would otherwise be counted SUBGROUP_SIZE times.
            xg_sum[idx] = acc / (float)SUBGROUP_SIZE;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#    endif
    }

    // Sweep the whole N range from inside one work-group rather than dispatching a separate
    // work-group per N block. The SLM staging above (TILE_M*K halfs, 32 KB at K=2048, plus the
    // xg_sum reduction) depends only on the token block, so dispatching N/(SUBGROUP_NUM*N_BLOCK)
    // work-groups over the same tokens repeated that staging 16x — which is what made TTFT at a
    // 1k prompt 10.3 s against 2.3 s for the unpack path. Total FLOPs are unchanged; only the
    // redundant staging goes away.
    const int n_per_wg = SUBGROUP_NUM * N_BLOCK;
    for (int n_super = 0; n_super < U2_GEMM_N; n_super += n_per_wg) {
    const int n_start = n_super + (int)get_local_id(2) * N_BLOCK;
    const int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + (size_t)n * K / 4;
        const __global MOE_SCALE_DT* S = scales + n;
#    if HAS_ZP && !U2_GEMM_ZP_SCALAR
        // n is even, so channels n and n+1 always share one packed byte: one load, two shifts.
        const __global uchar* Z = zps + n / 4;
        const int zshift = (n & 3) * 2;
#    endif
        // Accumulators are per-lane partial sums over this lane's K subset, one pair per token.
        float acc0[TILE_M];
        float acc1[TILE_M];
        unroll_for(int m = 0; m < TILE_M; m++) {
            acc0[m] = 0.0f;
            acc1[m] = 0.0f;
        }

        unroll_for(int gk = 0; gk < U2_GEMM_NUM_GK; gk++) {
            const int scale_offset = (gk * FAKE_GROUP_SIZE / U2_GEMM_GROUP_SIZE) * N;
            const half s0 = MOE_SCALE_AT(S, scale_offset, U2_GEMM_SCALE_GLOBAL);
            const half s1 = MOE_SCALE_AT(S, scale_offset + 1, U2_GEMM_SCALE_GLOBAL);
#    if HAS_ZP && U2_GEMM_ZP_SCALAR
            // INT2_SYM emits a single scalar zp shared by every expert, group and channel.
            const half z_hf0 = convert_half(((const __global MOE_ZP_SCALAR_DT*)zp_ptr)[0]);
            const half z_hf1 = z_hf0;
#    elif HAS_ZP
            // INT2_ASYM: one zp per (group, channel), indexed exactly like the scale above.
            const uchar zb = Z[(gk * FAKE_GROUP_SIZE / U2_GEMM_GROUP_SIZE) * N / 4];
            const half z_hf0 = convert_half((zb >> zshift) & 0x3);
            const half z_hf1 = convert_half((zb >> (zshift + 2)) & 0x3);
#    endif

            // --- Load and dequantise the weights ONCE for this (n, n+1, group) ---
            // w0[] / w1[] hold this lane's ELEMS_PER_LANE weights for channels n and n+1.
            half w0[ELEMS_PER_LANE];
            half w1[ELEMS_PER_LANE];
#    if ELEMS_PER_LANE == 4
            // One byte per lane covers 4 K-elements; block read is coalesced across the subgroup.
            const uchar b = intel_sub_group_block_read_uc(B + gk * FAKE_GROUP_SIZE / 4);
            const uchar b2 = intel_sub_group_block_read_uc(B + (K / 4) + gk * FAKE_GROUP_SIZE / 4);
            w0[0] = DEQUANT_2BIT(b, 0);
            w0[1] = DEQUANT_2BIT(b, 2);
            w0[2] = DEQUANT_2BIT(b, 4);
            w0[3] = DEQUANT_2BIT(b, 6);
            w1[0] = DEQUANT_2BIT(b2, 0);
            w1[1] = DEQUANT_2BIT(b2, 2);
            w1[2] = DEQUANT_2BIT(b2, 4);
            w1[3] = DEQUANT_2BIT(b2, 6);
#    elif ELEMS_PER_LANE == 2
            // Two lanes share one packed byte. Keep the per-lane scalar load: a subgroup block
            // read would make lanes >= FAKE_GROUP_SIZE/4 read past the tile and fault at the end
            // of the weights buffer (CL_OUT_OF_RESOURCES), same as in the GEMV.
            const uchar b = B[gk * FAKE_GROUP_SIZE / 4 + (id_local >> 1)];
            const uchar b2 = B[(K / 4) + gk * FAKE_GROUP_SIZE / 4 + (id_local >> 1)];
            const int wshift = (id_local & 1) * 4;
            w0[0] = DEQUANT_2BIT(b, wshift);
            w0[1] = DEQUANT_2BIT(b, wshift + 2);
            w1[0] = DEQUANT_2BIT(b2, wshift);
            w1[1] = DEQUANT_2BIT(b2, wshift + 2);
#    else  // ELEMS_PER_LANE == 1
            // Four lanes share one packed byte; same scalar-load rationale as above.
            const uchar b = B[gk * FAKE_GROUP_SIZE / 4 + (id_local >> 2)];
            const uchar b2 = B[(K / 4) + gk * FAKE_GROUP_SIZE / 4 + (id_local >> 2)];
            const int wshift = (id_local & 3) * 2;
            w0[0] = DEQUANT_2BIT(b, wshift);
            w1[0] = DEQUANT_2BIT(b2, wshift);
#    endif

            // --- Reuse those weights across every staged token: this is the entire win ---
            unroll_for(int m = 0; m < TILE_M; m++) {
                const __local half* xrow = x2 + m * K + gk * FAKE_GROUP_SIZE;
                half p0 = (half)0;
                half p1 = (half)0;
                unroll_for(int e = 0; e < ELEMS_PER_LANE; e++) {
                    const half av = xrow[id_local * ELEMS_PER_LANE + e];
                    p0 = fma(av, w0[e], p0);
                    p1 = fma(av, w1[e], p1);
                }
#    if HAS_ZP
                const float xs = xg_sum[m * U2_GEMM_NUM_GK + gk];
                acc0[m] += (convert_float(p0) - xs * convert_float(z_hf0)) * convert_float(s0);
                acc1[m] += (convert_float(p1) - xs * convert_float(z_hf1)) * convert_float(s1);
#    else
                acc0[m] += convert_float(p0) * convert_float(s0);
                acc1[m] += convert_float(p1) * convert_float(s1);
#    endif
            }
        }

        // Each lane held a partial over its own K subset; fold them across the subgroup.
        unroll_for(int m = 0; m < TILE_M; m++) {
            const float r0 = sub_group_reduce_add(acc0[m]);
            const float r1 = sub_group_reduce_add(acc1[m]);
            if (id_local == 0 && m < n_tokens) {
                __global half* out = c_ptr + (size_t)(token_start + m) * N;
                out[n] = convert_half(r0);
                out[n + 1] = convert_half(r1);
            }
        }
    }
    }  // n_super
}

#    endif  // U2_DPAS_ENABLE
#endif
