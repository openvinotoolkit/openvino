// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

// Implicit-GEMM 1D convolution, C = A * B with
//
//   A: weights, M x K, M = OC,               K = IC * FILTER_LEN
//   B: im2col,  K x N, N = BATCH * OUT_LEN
//
// Neither is reordered or materialized: a K index decomposes into
// (input feature, filter position) and an N index into (batch, output position),
// so the input element a (k, n) pair refers to is
//
//   input[batch][ic][n_pos * STRIDE + fpos * DILATION - PAD_BEGIN]
//
// resolved while staging tiles. Out-of-range positions are zero-filled there, so
// the accumulation loop is branch-free. See convolution_kernel_1d_small_ic_gemm.h.

// The host picks the long spatial axis and sets LONG_AXIS_IS_X, so these stay
// correct whether or not the plugin's XY swap applied.
#if LONG_AXIS_IS_X
    #define INPUT_AT(b, ic, pos)  INPUT0_GET_INDEX(b, ic, 0, pos)
    #define OUTPUT_AT(b, oc, pos) OUTPUT_GET_INDEX(b, oc, 0, pos)
    #define WEIGHTS_AT(oc, ic, fpos) GET_FILTER_INDEX(FILTER, 0, oc, ic, 0, fpos)
#else
    #define INPUT_AT(b, ic, pos)  INPUT0_GET_INDEX(b, ic, pos, 0)
    #define OUTPUT_AT(b, oc, pos) OUTPUT_GET_INDEX(b, oc, pos, 0)
    #define WEIGHTS_AT(oc, ic, fpos) GET_FILTER_INDEX(FILTER, 0, oc, ic, fpos, 0)
#endif

// A work-group computes a TILE_M x TILE_N tile but is only one sub-group wide:
// each lane owns N_PER_LANE columns and all TILE_M rows in registers.
#define WG_SIZE (SIMD)

// Vector load of B_VEC consecutive input elements, only valid on the B_IN_BOUNDS
// path where the host has proved a run of filter positions is contiguous.
#define B_VEC_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, B_VEC)
#define B_VLOAD(ptr) CAT(vload, B_VEC)(0, ptr)

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
KERNEL(convolution_gpu_1d_small_ic_gemm)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE *conv_input,
    __global OUTPUT_TYPE *output,
    const __global FILTER_TYPE *weights
#if BIAS_TERM
    , const __global BIAS_TYPE *biases
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    , const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    , const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp
#endif
#if COMPENSATION_TERM
    , const __global COMPENSATION_TYPE *comp
#endif
)
{
    const uint lid = get_local_id(0);

    // Tile origin in the GEMM index space.
    const uint n_base = (uint)get_group_id(0) * TILE_N;
    const uint m_base = (uint)get_group_id(1) * TILE_M;

    // One K slice of A. B is per-lane and stays in registers.
    __local ACCUMULATOR_TYPE a_tile[TILE_M][TILE_K];

    // Both extents are compile-time constants and every access below is at a
    // constant index, so this stays in registers instead of spilling.
    ACCUMULATOR_TYPE acc[N_PER_LANE][TILE_M];
    __attribute__((opencl_unroll_hint))
    for (uint j = 0; j < N_PER_LANE; ++j)
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < TILE_M; ++i)
            acc[j][i] = (ACCUMULATOR_TYPE)0;

    // This lane's output columns. SIMD apart rather than adjacent, so that at a
    // given j the sub-group covers consecutive columns and stays coalesced.
    uint n_batch[N_PER_LANE];
    uint n_pos[N_PER_LANE];
    bool n_valid[N_PER_LANE];
    __attribute__((opencl_unroll_hint))
    for (uint j = 0; j < N_PER_LANE; ++j) {
        const uint n = n_base + j * SIMD + lid;
        n_batch[j] = n / OUT_LEN;
        n_pos[j] = n % OUT_LEN;
        n_valid[j] = (n < GEMM_N);
    }

    for (uint k_base = 0; k_base < GEMM_K; k_base += TILE_K) {
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Stage A (weights): TILE_M x TILE_K, strided over the WG. ---
        for (uint idx = lid; idx < TILE_M * TILE_K; idx += WG_SIZE) {
            const uint am = idx / TILE_K;
            const uint ak = idx % TILE_K;
            const uint m = m_base + am;
            const uint k = k_base + ak;

            ACCUMULATOR_TYPE w = (ACCUMULATOR_TYPE)0;
            if (m < GEMM_M && k < GEMM_K) {
                // k -> (input feature, filter position)
                const uint k_ic = k / FILTER_LEN;
                const uint k_fpos = k % FILTER_LEN;
                const uint w_idx = WEIGHTS_AT(m, k_ic, k_fpos);
                w = TO_ACCUMULATOR_TYPE(weights[w_idx]);
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
                w -= TO_ACCUMULATOR_TYPE(weights_zp[m]);
#endif
            }
            a_tile[am][ak] = w;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Stage B (im2col column block) and accumulate. ---
        // Staging and accumulation are fused so B lives in a few registers rather
        // than a dynamically indexed TILE_K array, which would spill. Loop order is
        // (k outer, column inner) so each SLM read of A is reused by all N_PER_LANE
        // columns - the whole reason for the 2D tile.
#if B_IN_BOUNDS
        // Fast path: every position is in range and a K run is contiguous, so
        // TILE_K scalar gathers become a handful of wide loads. The host proves this
        // for the whole iteration space; see GetJitConstants.
        {
            const uint k_ic = k_base / FILTER_LEN;
            const uint k_fpos = k_base % FILTER_LEN;

            const __global INPUT0_TYPE* src[N_PER_LANE];
            __attribute__((opencl_unroll_hint))
            for (uint j = 0; j < N_PER_LANE; ++j) {
                const uint pos = n_pos[j] * STRIDE_L + k_fpos * DILATION_L - PAD_BEGIN_L;
                src[j] = conv_input + INPUT_AT(n_batch[j], k_ic, pos);
            }
#if ASYMMETRIC_DATA_QUANTIZATION
            const ACCUMULATOR_TYPE azp = TO_ACCUMULATOR_TYPE(activations_zp[k_ic]);
#endif
            for (uint bk = 0; bk < TILE_K; bk += B_VEC) {
                B_VEC_TYPE raw[N_PER_LANE];
                __attribute__((opencl_unroll_hint))
                for (uint j = 0; j < N_PER_LANE; ++j)
                    raw[j] = B_VLOAD(src[j] + bk);

                __attribute__((opencl_unroll_hint))
                for (uint e = 0; e < B_VEC; ++e) {
                    ACCUMULATOR_TYPE bv[N_PER_LANE];
                    __attribute__((opencl_unroll_hint))
                    for (uint j = 0; j < N_PER_LANE; ++j) {
                        bv[j] = TO_ACCUMULATOR_TYPE(raw[j][e]);
#if ASYMMETRIC_DATA_QUANTIZATION
                        bv[j] -= azp;
#endif
                    }

                    __attribute__((opencl_unroll_hint))
                    for (uint i = 0; i < TILE_M; ++i) {
                        const ACCUMULATOR_TYPE av = a_tile[i][bk + e];
                        __attribute__((opencl_unroll_hint))
                        for (uint j = 0; j < N_PER_LANE; ++j)
                            acc[j][i] = fma(av, bv[j], acc[j][i]);
                    }
                }
            }
        }
#else
        // General path: guard every element. Used when padding, dilation or a ragged
        // tile can put a position out of range.
        for (uint bk = 0; bk < TILE_K; ++bk) {
            const uint k = k_base + bk;

            ACCUMULATOR_TYPE bv[N_PER_LANE];
            __attribute__((opencl_unroll_hint))
            for (uint j = 0; j < N_PER_LANE; ++j) {
                bv[j] = (ACCUMULATOR_TYPE)0;
                if (n_valid[j] && k < GEMM_K) {
                    const uint k_ic = k / FILTER_LEN;
                    const uint k_fpos = k % FILTER_LEN;
                    const int pos = (int)(n_pos[j] * STRIDE_L + k_fpos * DILATION_L) - (int)PAD_BEGIN_L;
                    if (pos >= 0 && pos < (int)IN_LEN) {
                        const uint in_idx = INPUT_AT(n_batch[j], k_ic, (uint)pos);
                        bv[j] = TO_ACCUMULATOR_TYPE(conv_input[in_idx]);
#if ASYMMETRIC_DATA_QUANTIZATION
                        bv[j] -= TO_ACCUMULATOR_TYPE(activations_zp[k_ic]);
#endif
                    }
                }
            }

            __attribute__((opencl_unroll_hint))
            for (uint i = 0; i < TILE_M; ++i) {
                const ACCUMULATOR_TYPE av = a_tile[i][bk];
                __attribute__((opencl_unroll_hint))
                for (uint j = 0; j < N_PER_LANE; ++j)
                    acc[j][i] = fma(av, bv[j], acc[j][i]);
            }
        }
#endif
    }

    // --- Epilogue: bias, activation, store. ---
    // M outermost so that, for a fixed i, the lanes at a given j write consecutive
    // output elements.
    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < TILE_M; ++i) {
        const uint m = m_base + i;
        if (m >= GEMM_M)
            continue;

        __attribute__((opencl_unroll_hint))
        for (uint j = 0; j < N_PER_LANE; ++j) {
            if (!n_valid[j])
                continue;

            ACTIVATION_TYPE res = TO_ACTIVATION_TYPE(acc[j][i]);
            // No compensation term on purpose: this kernel already subtracted the
            // activation zero point per element while staging B, so applying the
            // precomputed -sum(azp * w) would double-count.
#if BIAS_TERM
            res += TO_ACTIVATION_TYPE(biases[m]);
#endif
            const uint dst = OUTPUT_AT(n_batch[j], m, n_pos[j]);
            output[dst] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(res, ACTIVATION_PARAMS_TYPED));
        }
    }
}

#undef WG_SIZE
#undef INPUT_AT
#undef OUTPUT_AT
#undef WEIGHTS_AT
