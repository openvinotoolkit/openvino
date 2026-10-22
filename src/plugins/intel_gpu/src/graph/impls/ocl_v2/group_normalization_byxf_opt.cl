// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// byxf keeps the features of a pixel contiguous: a work-item owns VEC features of one batch and walks a chunk of pixels.
#define ACC_VEC_TYPE    MAKE_VECTOR_TYPE(float, VEC)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC)
#if VEC == 1
#    define LOAD_VEC(ptr, offset)       convert_float((ptr)[offset])
#    define STORE_VEC(ptr, offset, val) (ptr)[offset] = (val)
#    define TO_OUTPUT_VEC_TYPE(val)     TO_OUTPUT_TYPE(val)
#    define VEC_ELEM(val, i)            (val)
#else
#    define LOAD_VEC(ptr, offset)       CAT(convert_float, VEC)(CAT(vload, VEC)(0, (ptr) + (offset)))
#    define STORE_VEC(ptr, offset, val) CAT(vstore, VEC)(val, 0, (ptr) + (offset))
#    define TO_OUTPUT_VEC_TYPE(val)     CAT(convert_, OUTPUT_VEC_TYPE)(val)
#    define VEC_ELEM(val, i)            (val)[i]
#endif

#if GROUP_NORM_KERNEL_CALC_STATS
// Loads in flight per work-item (2 and 8 were slower).
#    define STATS_UNROLL 4
KERNEL(calc_stats)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input, __global float* sum, __global float* sq_sum) {
    const uint fv = get_global_id(0);
    if (fv >= FV) {
        return;
    }
    const uint chunk = get_global_id(1);
    const uint num_chunks = get_global_size(1);
    const uint b = get_global_id(2);
    const uint f = fv * VEC;

    const uint pixels = INPUT0_SIZE_Y * INPUT0_SIZE_X;

    // Sums of d = x - x[b, f, 0, 0]: E[d^2] - E[d]^2 does not cancel when |mean| >> std.
    const ACC_VEC_TYPE pivot = LOAD_VEC(input, INPUT0_GET_INDEX(b, f, 0, 0));
    ACC_VEC_TYPE s1 = 0.0f;
    ACC_VEC_TYPE s2 = 0.0f;
    // Chunk c takes pixels c, c + num_chunks, ...: all work-items read neighbouring pixels, so the tensor streams front to back.
    uint p = chunk;
    // Without x padding the pixels of a batch are equally spaced, so several loads can be in flight per work-item.
    if (INPUT0_Y_PITCH == INPUT0_X_PITCH * INPUT0_SIZE_X) {
        const uint step = num_chunks * INPUT0_X_PITCH;
        uint offset = INPUT0_GET_INDEX(b, f, 0, 0) + p * INPUT0_X_PITCH;
        for (; p + (STATS_UNROLL - 1) * num_chunks < pixels; p += STATS_UNROLL * num_chunks) {
            ACC_VEC_TYPE d[STATS_UNROLL];
            unroll_for(uint i = 0; i < STATS_UNROLL; ++i) {
                d[i] = LOAD_VEC(input, offset + i * step) - pivot;
            }
            unroll_for(uint i = 0; i < STATS_UNROLL; ++i) {
                s1 += d[i];
                s2 = mad(d[i], d[i], s2);
            }
            offset += STATS_UNROLL * step;
        }
    }
    for (; p < pixels; p += num_chunks) {
        const uint y = p / INPUT0_SIZE_X;
        const ACC_VEC_TYPE d = LOAD_VEC(input, INPUT0_GET_INDEX(b, f, y, p - y * INPUT0_SIZE_X)) - pivot;
        s1 += d;
        s2 = mad(d, d, s2);
    }

    const uint out_offset = (chunk * INPUT0_BATCH_NUM + b) * INPUT0_FEATURE_NUM + f;
    STORE_VEC(sum, out_offset, s1);
    STORE_VEC(sq_sum, out_offset, s2);
}
#elif GROUP_NORM_KERNEL_FINALIZE_STATS
KERNEL(finalize_stats)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
 const __global INPUT1_TYPE* scale,
 const __global INPUT2_TYPE* bias,
 const __global float* sum,
 const __global float* sq_sum,
 __global float* scale_shift) {
    // Work-group = (features of the group) x (chunks): the partial sums are read in parallel, not in a GROUP_SIZE loop.
    const uint lane = get_local_id(0);
    const uint num_lanes = get_local_size(0);
    const uint chunk = get_local_id(1);
    const uint num_chunks = get_local_size(1);
    const uint group = get_group_id(0);
    const uint b = get_global_id(2);
    const uint f0 = group * GROUP_SIZE;
    const uint lid = chunk * num_lanes + lane;
    const uint wg_size = num_lanes * num_chunks;
    const float pixels = INPUT0_SIZE_Y * INPUT0_SIZE_X;
    const float group_elements = pixels * GROUP_SIZE;

    __local float pivot[GROUP_SIZE];
    for (uint i = lid; i < GROUP_SIZE; i += wg_size) {
        pivot[i] = convert_float(input[INPUT0_GET_INDEX(b, f0 + i, 0, 0)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float pivot_sum = 0.0f;
    for (uint i = 0; i < GROUP_SIZE; ++i) {
        pivot_sum += pivot[i];
    }
    // The pivot terms are added once, after the work-group reductions.
    const uint row = (chunk * INPUT0_BATCH_NUM + b) * INPUT0_FEATURE_NUM + f0;
    float s1 = 0.0f;
    for (uint i = lane; i < GROUP_SIZE; i += num_lanes) {
        s1 += sum[row + i];
    }
    const float mean = pivot_sum / GROUP_SIZE + work_group_reduce_add(s1) / group_elements;

    // sum_group (x - mean)^2 = sum_f [S2_f + 2 * (K_f - mean) * S1_f + pixels * (K_f - mean)^2]
    float m2 = 0.0f;
    for (uint i = lane; i < GROUP_SIZE; i += num_lanes) {
        m2 += sq_sum[row + i] + 2.0f * (pivot[i] - mean) * sum[row + i];
    }
    float pivot_dev = 0.0f;
    for (uint i = 0; i < GROUP_SIZE; ++i) {
        const float d = pivot[i] - mean;
        pivot_dev = mad(d, d, pivot_dev);
    }
    const float variance = fmax((work_group_reduce_add(m2) + pixels * pivot_dev) / group_elements, 0.0f);
    const float inv_std = rsqrt(variance + EPSILON);

    for (uint i = lid; i < GROUP_SIZE; i += wg_size) {
        const uint f = f0 + i;
        const float a = convert_float(scale[f]) * inv_std;
        scale_shift[b * INPUT0_FEATURE_NUM + f] = a;
        scale_shift[(INPUT0_BATCH_NUM + b) * INPUT0_FEATURE_NUM + f] = convert_float(bias[f]) - mean * a;
    }
}
#elif GROUP_NORM_KERNEL_NORMALIZE
#    if HAS_FUSED_OPS
// Fused ops take (b, f, y, x) of each element, so y and x must be current at every use.
#        define NORMALIZE_STORE(v, offset)                               \
            {                                                            \
                OUTPUT_VEC_TYPE res;                                     \
                unroll_for(uint j = 0; j < VEC; ++j) {                   \
                    const uint f = f0 + j;                               \
                    const float normalized = VEC_ELEM(v, j);             \
                    FUSED_OPS;                                           \
                    VEC_ELEM(res, j) = TO_OUTPUT_TYPE(FUSED_OPS_RESULT); \
                }                                                        \
                STORE_VEC(output, offset, res);                          \
            }
#    else
#        define NORMALIZE_STORE(v, offset) STORE_VEC(output, offset, TO_OUTPUT_VEC_TYPE(v))
#    endif
#    define NORM_UNROLL 4
KERNEL(normalize)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
 const __global float* scale_shift,
#    if HAS_FUSED_OPS_DECLS
 FUSED_OPS_DECLS,
#    endif
 __global OUTPUT_TYPE* restrict output) {
    const uint fv = get_global_id(0);
    if (fv >= FV) {
        return;
    }
    const uint num_chunks = get_global_size(1);
    // Back to front: calc_stats streamed the tensor front to back, so its tail may still be in the cache.
    const uint chunk = num_chunks - 1 - get_global_id(1);
    const uint b = get_global_id(2);
    const uint f0 = fv * VEC;

    // y = x * A + C, A = scale * inv_std and C = bias - mean * A per (b, f)
    const ACC_VEC_TYPE scale_vec = LOAD_VEC(scale_shift, b * OUTPUT_FEATURE_NUM + f0);
    const ACC_VEC_TYPE shift_vec = LOAD_VEC(scale_shift, (OUTPUT_BATCH_NUM + b) * OUTPUT_FEATURE_NUM + f0);

    const uint pixels = OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const uint chunk_size = CEIL_DIV(pixels, num_chunks);
    uint p = chunk * chunk_size;
    const uint p_end = min(pixels, p + chunk_size);
    uint y = p / OUTPUT_SIZE_X;
    uint x = p - y * OUTPUT_SIZE_X;
    uint in_offset = INPUT0_GET_INDEX(b, f0, y, x);
    uint out_offset = OUTPUT_GET_INDEX(b, f0, y, x);
    uint n = p < p_end ? p_end - p : 0;
    const bool linear = INPUT0_Y_PITCH == INPUT0_X_PITCH * INPUT0_SIZE_X && OUTPUT_Y_PITCH == OUTPUT_X_PITCH * OUTPUT_SIZE_X;
    if (linear) {
        for (; n >= NORM_UNROLL; n -= NORM_UNROLL) {
            ACC_VEC_TYPE v[NORM_UNROLL];
            unroll_for(uint i = 0; i < NORM_UNROLL; ++i) {
                v[i] = mad(LOAD_VEC(input, in_offset + i * INPUT0_X_PITCH), scale_vec, shift_vec);
            }
            unroll_for(uint i = 0; i < NORM_UNROLL; ++i) {
                NORMALIZE_STORE(v[i], out_offset + i * OUTPUT_X_PITCH);
                if (++x == OUTPUT_SIZE_X) {
                    x = 0;
                    ++y;
                }
            }
            in_offset += NORM_UNROLL * INPUT0_X_PITCH;
            out_offset += NORM_UNROLL * OUTPUT_X_PITCH;
        }
    }
    for (; n > 0; --n) {
        const ACC_VEC_TYPE v = mad(LOAD_VEC(input, in_offset), scale_vec, shift_vec);
        NORMALIZE_STORE(v, out_offset);
        in_offset += INPUT0_X_PITCH;
        out_offset += OUTPUT_X_PITCH;
        if (++x == OUTPUT_SIZE_X) {
            x = 0;
            ++y;
            if (!linear) {
                in_offset = INPUT0_GET_INDEX(b, f0, y, 0);
                out_offset = OUTPUT_GET_INDEX(b, f0, y, 0);
            }
        }
    }
}
#endif
