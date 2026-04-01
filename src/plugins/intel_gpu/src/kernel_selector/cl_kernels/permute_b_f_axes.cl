// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Permute kernel for B <-> F axis swap (order [1,0,2,3] and higher-dim equivalents).
// X is contiguous before and after the swap, so no SLM transpose is needed.
// Each work item vectorizes along X with vload/vstore, processes Y_BLOCK Y-rows
// to improve memory-level parallelism, and loops over F internally.
//
// GWS: (ceil(X / VEC_WIDTH), ceil(Y / Y_BLOCK) [* Z [* W]], B)

#include "include/batch_headers/fetch_data.cl"

KERNEL(permute_b_f_axes)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x_t = get_global_id(0);

#if INPUT0_DIMS == 4
    const uint y_base = get_global_id(1) * Y_BLOCK;
#elif INPUT0_DIMS == 5
    const uint z = get_global_id(1) / Y_TILES_Y;
    const uint y_base = (get_global_id(1) % Y_TILES_Y) * Y_BLOCK;
#elif INPUT0_DIMS == 6
    const uint w = get_global_id(1) / (INPUT0_SIZE_Z * Y_TILES_Y);
    const uint z = (get_global_id(1) / Y_TILES_Y) % INPUT0_SIZE_Z;
    const uint y_base = (get_global_id(1) % Y_TILES_Y) * Y_BLOCK;
#endif

    const uint b = get_global_id(2);
    const uint x_base = x_t * VEC_WIDTH;

    for (uint f = 0; f < INPUT0_FEATURE_NUM; ++f) {

#if X_REMAINDER_SIZE > 0
        if (x_t == X_TILES) {
            __attribute__((opencl_unroll_hint(Y_BLOCK)))
            for (uint yb = 0; yb < Y_BLOCK; ++yb) {
                const uint y = y_base + yb;
                for (uint i = 0; i < X_REMAINDER_SIZE; ++i) {
                    const uint x = x_base + i;
#if INPUT0_DIMS == 4
                    const uint in_idx = INPUT0_GET_INDEX(b, f, y, x);
                    const uint out_idx = OUTPUT_GET_INDEX(f, b, y, x);
#elif INPUT0_DIMS == 5
                    const uint in_idx = INPUT0_GET_INDEX(b, f, z, y, x);
                    const uint out_idx = OUTPUT_GET_INDEX(f, b, z, y, x);
#elif INPUT0_DIMS == 6
                    const uint in_idx = INPUT0_GET_INDEX(b, f, w, z, y, x);
                    const uint out_idx = OUTPUT_GET_INDEX(f, b, w, z, y, x);
#endif
                    INPUT0_TYPE val = input[in_idx];
#if HAS_FUSED_OPS
                    INPUT0_TYPE input_var = val;
                    FUSED_OPS;
                    output[out_idx] = FUSED_OPS_RESULT;
#else
                    output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
                }
            }
            continue;
        }
#endif  // X_REMAINDER_SIZE > 0

        // Load Y_BLOCK rows then store, to improve memory-level parallelism.
        INPUTVTYPE vals[Y_BLOCK];
        uint out_idxs[Y_BLOCK];

        __attribute__((opencl_unroll_hint(Y_BLOCK)))
        for (uint yb = 0; yb < Y_BLOCK; ++yb) {
            const uint y = y_base + yb;
#if INPUT0_DIMS == 4
            vals[yb] = CAT(vload, VEC_WIDTH)(0, input + INPUT0_GET_INDEX(b, f, y, x_base));
            out_idxs[yb] = OUTPUT_GET_INDEX(f, b, y, x_base);
#elif INPUT0_DIMS == 5
            vals[yb] = CAT(vload, VEC_WIDTH)(0, input + INPUT0_GET_INDEX(b, f, z, y, x_base));
            out_idxs[yb] = OUTPUT_GET_INDEX(f, b, z, y, x_base);
#elif INPUT0_DIMS == 6
            vals[yb] = CAT(vload, VEC_WIDTH)(0, input + INPUT0_GET_INDEX(b, f, w, z, y, x_base));
            out_idxs[yb] = OUTPUT_GET_INDEX(f, b, w, z, y, x_base);
#endif
        }

        __attribute__((opencl_unroll_hint(Y_BLOCK)))
        for (uint yb = 0; yb < Y_BLOCK; ++yb) {
#if HAS_FUSED_OPS
            const uint y = y_base + yb;
            OUTPUTVTYPE out_vals;
            __attribute__((opencl_unroll_hint(VEC_WIDTH)))
            for (uint i = 0; i < VEC_WIDTH; ++i) {
                INPUT0_TYPE input_var = vals[yb][i];
                FUSED_OPS;
                out_vals[i] = FUSED_OPS_RESULT;
            }
            CAT(vstore, VEC_WIDTH)(out_vals, 0, output + out_idxs[yb]);
#else
            OUTPUTVTYPE out_vals = ACTIVATION(CAT(convert_, OUTPUTVTYPE)(vals[yb]), ACTIVATION_PARAMS);
            CAT(vstore, VEC_WIDTH)(out_vals, 0, output + out_idxs[yb]);
#endif
        }
    }  // for f
}
