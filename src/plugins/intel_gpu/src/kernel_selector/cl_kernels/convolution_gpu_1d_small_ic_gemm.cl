// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

// Implicit-GEMM 1D convolution for a small input feature count and a large tap
// count. See convolution_kernel_1d_small_ic_gemm.h for the rationale.
//
// The convolution is treated as C = A * B with
//
//   A: weights, M x K, M = OC,               K = IC * TAPS
//   B: im2col,  K x N, N = BATCH * OUT_LEN
//
// Neither is reordered or materialized. A K index decomposes into
// (input feature, tap) and an N index into (batch, output position); the input
// element a (k, n) pair refers to is
//
//   input[batch][ic][n_pos * STRIDE + tap * DILATION - PAD_BEGIN]
//
// computed while staging tiles into SLM. Out-of-range positions are zero-filled
// there, so the accumulation loop is branch-free. Validate() accepts IC == 1 only
// today, but that is selection policy: the decomposition above is correct for any
// IC, so nothing here changes if max_input_features is raised.
//
// Jit constants supplied by the host:
//   TAPS, OUT_LEN, GEMM_M, GEMM_N, GEMM_K
//   TILE_M, TILE_N, TILE_K, SIMD, N_PER_LANE
//   STRIDE_L, DILATION_L, PAD_BEGIN_L, IN_LEN
//   ACCUMULATOR_TYPE, ACTIVATION_TYPE
//   B_IN_BOUNDS, B_VEC (see below)

// The long spatial axis is X or Y depending on whether the plugin's XY swap
// applied. The host picks the long one and sets LONG_AXIS_IS_X accordingly, so
// the index macros below stay correct either way.
#if LONG_AXIS_IS_X
    #define INPUT_AT(b, ic, pos)  INPUT0_GET_INDEX(b, ic, 0, pos)
    #define OUTPUT_AT(b, oc, pos) OUTPUT_GET_INDEX(b, oc, 0, pos)
    #define WEIGHTS_AT(oc, ic, tap) GET_FILTER_INDEX(FILTER, 0, oc, ic, 0, tap)
#else
    #define INPUT_AT(b, ic, pos)  INPUT0_GET_INDEX(b, ic, pos, 0)
    #define OUTPUT_AT(b, oc, pos) OUTPUT_GET_INDEX(b, oc, pos, 0)
    #define WEIGHTS_AT(oc, ic, tap) GET_FILTER_INDEX(FILTER, 0, oc, ic, tap, 0)
#endif

// One work-group computes a TILE_M x TILE_N tile but is only one sub-group (SIMD
// work items) wide: each lane owns N_PER_LANE output columns and all TILE_M rows
// in registers. TILE_N == SIMD * N_PER_LANE.
//
// N_PER_LANE > 1 is what fixes the inner loop's memory-to-arithmetic ratio: since
// a_tile[i][k] does not depend on which output column is accumulated, one SLM read
// of A feeds N_PER_LANE FMAs instead of one.
#define WG_SIZE (SIMD)

// Vector load of B_VEC consecutive input elements. Only used on the B_IN_BOUNDS
// path, where the host has checked long-axis pitch 1 and DILATION_L == 1, so a run
// of consecutive taps is contiguous.
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

    // SLM staging buffer for one K slice of A. B is per-lane and stays in
    // registers - see B_VLOAD above.
    __local ACCUMULATOR_TYPE a_tile[TILE_M][TILE_K];

    // TILE_M rows x N_PER_LANE columns per lane. Both extents are compile-time
    // constants and every access below is at a constant index, so this stays in the
    // register file instead of spilling to scratch.
    ACCUMULATOR_TYPE acc[N_PER_LANE][TILE_M];
    __attribute__((opencl_unroll_hint))
    for (uint j = 0; j < N_PER_LANE; ++j)
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < TILE_M; ++i)
            acc[j][i] = (ACCUMULATOR_TYPE)0;

    // This lane's output columns, i.e. its (batch, output position) pairs. They are
    // SIMD apart rather than adjacent so that at a given j the sub-group's lanes
    // cover consecutive columns, keeping the B loads and the stores coalesced.
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
                // k -> (input feature, tap)
                const uint k_ic = k / TAPS;
                const uint k_tap = k % TAPS;
                const uint w_idx = WEIGHTS_AT(m, k_ic, k_tap);
                w = TO_ACCUMULATOR_TYPE(weights[w_idx]);
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
                w -= TO_ACCUMULATOR_TYPE(weights_zp[m]);
#endif
            }
            a_tile[am][ak] = w;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Stage B (im2col column block) and accumulate. ---
        // A lane is the only reader of its own K slices, so B never goes to SLM.
        // Staging and accumulation are fused so B lives in a few registers rather
        // than a TILE_K array, which would be dynamically indexed and spill.
        //
        // Loop order is (k outer, column inner) so each SLM read of a_tile[i][k] is
        // reused by all N_PER_LANE columns - the whole reason for the 2D tile.
        // Reversing the loops would read A N_PER_LANE times as often.
#if B_IN_BOUNDS
        // Fast path: every position this tile touches is in range and a K run is
        // contiguous, so B is read with unguarded vector loads - TILE_K scalar
        // gathers become a handful of wide ones. The host proves all of that for the
        // whole iteration space before setting B_IN_BOUNDS; see GetJitConstants.
        {
            const uint k_ic = k_base / TAPS;
            const uint k_tap = k_base % TAPS;

            const __global INPUT0_TYPE* src[N_PER_LANE];
            __attribute__((opencl_unroll_hint))
            for (uint j = 0; j < N_PER_LANE; ++j) {
                const uint pos = n_pos[j] * STRIDE_L + k_tap * DILATION_L - PAD_BEGIN_L;
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
        // General path: guard every element. Used when padding, dilation, a ragged
        // tile or an indivisible TAPS can put a position out of range.
        for (uint bk = 0; bk < TILE_K; ++bk) {
            const uint k = k_base + bk;

            ACCUMULATOR_TYPE bv[N_PER_LANE];
            __attribute__((opencl_unroll_hint))
            for (uint j = 0; j < N_PER_LANE; ++j) {
                bv[j] = (ACCUMULATOR_TYPE)0;
                if (n_valid[j] && k < GEMM_K) {
                    const uint k_ic = k / TAPS;
                    const uint k_tap = k % TAPS;
                    const int pos = (int)(n_pos[j] * STRIDE_L + k_tap * DILATION_L) - (int)PAD_BEGIN_L;
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
    // The M loop is outermost so that, for a fixed i, the lanes at a given j write
    // consecutive output elements.
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
            // No compensation term on purpose: it is the precomputed
            // -sum(azp * w) for kernels that cannot subtract the activation zero
            // point per element, but this one already did while staging B, so
            // applying it would double-count. convolution_gpu_ref also takes the
            // argument and ignores it.
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
