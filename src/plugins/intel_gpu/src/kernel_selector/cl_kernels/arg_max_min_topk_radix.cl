// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Radix/Histogram Select + Bitonic Sort TopK
//
// Algorithm:
//   Phase 0: Read input ONCE -> cache sortable keys in global buffer
//   Phase 1: Two-level histogram in SLM to find the K-th threshold value
//   Phase 2: Gather top-K elements directly into SLM
//   Phase 3: Bitonic sort in SLM (45 barriers for K<=512 vs 3000+ for iterative)
//   Phase 4: Write sorted results to output
//
// Design principles:
//   - Read input exactly ONCE to avoid GPU cache inconsistency
//   - Single global buffer for sortable keys (N * 4 bytes per operation)
//   - SLM bitonic sort for Phase 3 (fast, no global memory for sorting)
//   - Total SLM: ~5KB (histogram 1KB + sort buffers 4KB)
//

#include "include/fetch_utils.cl"

#ifdef BATCH_AXIS
    #define VALUES_NUM INPUT0_BATCH_NUM
    #define AXIS 0
#endif
#ifdef FEATURE_AXIS
    #define VALUES_NUM INPUT0_FEATURE_NUM
    #define AXIS 1
#endif
#ifdef Z_AXIS
    #define VALUES_NUM INPUT0_SIZE_Z
    #define AXIS 2
#endif
#ifdef Y_AXIS
    #define VALUES_NUM INPUT0_SIZE_Y
    #define AXIS 3
#endif
#ifdef X_AXIS
    #define VALUES_NUM INPUT0_SIZE_X
    #define AXIS 4
#endif

#ifdef MAX_OUT
    #define COMPARE_SIGN >
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN <
    #define INPUT0_FILL_VAL INPUT0_VAL_MAX
#endif

#ifndef WG_SIZE
    #define WG_SIZE 256
#endif

#ifndef PADDED_K
    #define PADDED_K 512
#endif

#define NUM_BUCKETS 256

inline void FUNC(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_ARG
                                        const uint output_idx,
                                        uint* indices)
{
#ifdef BATCH_AXIS
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[1] = out_first_dim; indices[2] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef FEATURE_AXIS
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[0] = out_first_dim; indices[2] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef Z_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef Y_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Z;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[2] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef X_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_Y);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y) % INPUT0_FEATURE_NUM;
    const uint out_third_dim = output_idx / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_Y;
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[2] = out_third_dim; indices[3] = out_fourth_dim;
#endif
}

// Convert f16 to a sortable uint16:
// For positive f16: the bit pattern is already sortable (larger value = larger uint)
// For negative f16: flip all bits
// This gives a monotonically increasing uint16 mapping
inline uint FUNC(f16_to_sortable)(INPUT0_TYPE val) {
    // Use as_ushort to get the bit pattern of f16
    ushort bits = as_ushort(convert_half(val));
    // If sign bit is set (negative), flip all bits
    // If sign bit is clear (positive), flip only sign bit
    // This maps f16 range to 0..65535 monotonically
    ushort mask = (bits >> 15) ? (ushort)0xFFFF : (ushort)0x8000;
    return (uint)(bits ^ mask);
}

inline INPUT0_TYPE FUNC(sortable_to_f16)(uint sortable) {
    ushort bits = (ushort)sortable;
    ushort mask = (bits & 0x8000) ? (ushort)0x8000 : (ushort)0xFFFF;
    bits = bits ^ mask;
    return convert_float(as_half(bits));
}

REQD_SUB_GROUP_SIZE(16)
KERNEL(arg_max_min_topk_radix)(
    const __global INPUT0_TYPE* input
    ,__global OUTPUT_TYPE* output
#ifdef OUTPUT1_TYPE
    ,__global OUTPUT1_TYPE* second_output
#endif
    ,__global uint* sortable_buf            // Cached sortable keys: VALUES_NUM per operation
)
{
    const uint lid = (uint)get_local_id(0);
    const uint output_idx = (uint)get_group_id(0);

    uint base_indices[] = { 0, 0, 0, 0, 0 };
    if (OPERATION_NUM > 1) {
        FUNC_CALL(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_TENSOR output_idx, base_indices);
    }

    // Global buffer pointer for this operation's sortable keys
    __global uint* my_sortable = sortable_buf + output_idx * VALUES_NUM;

    // ============================================================
    // SLM declarations (all at outermost kernel scope)
    // Total SLM: 256*4 + 512*4 + 512*4 + ~20 ~= 5KB
    // ============================================================
    __local uint histogram[NUM_BUCKETS];        // 1KB
    __local uint sort_keys[PADDED_K];           // sortable uint keys for bitonic sort
    __local uint sort_idxs[PADDED_K];           // original indices
    __local uint threshold_bucket;
    __local uint count_above;
    __local uint fine_threshold;
    __local uint total_above_threshold;
    __local uint gather_count;

    // ============================================================
    // Phase 0: Read input ONCE, convert to sortable keys, cache in global buffer
    // Also compute coarse histogram in the same pass
    // ============================================================
    if (lid < NUM_BUCKETS) {
        histogram[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = lid; i < VALUES_NUM; i += WG_SIZE) {
        base_indices[AXIS] = i;
        INPUT0_TYPE val = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR
            base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4])];
        uint sortable = FUNC_CALL(f16_to_sortable)(val);
        my_sortable[i] = sortable;
        uint bucket = sortable >> 8;
        atomic_add(&histogram[bucket], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Find coarse bucket containing K-th element
    if (lid == 0) {
        uint cumulative = 0;
#ifdef MAX_OUT
        for (int b = NUM_BUCKETS - 1; b >= 0; b--) {
            cumulative += histogram[b];
            if (cumulative >= TOP_K) {
                threshold_bucket = (uint)b;
                count_above = cumulative - histogram[b];
                break;
            }
        }
#else
        for (uint b = 0; b < NUM_BUCKETS; b++) {
            cumulative += histogram[b];
            if (cumulative >= TOP_K) {
                threshold_bucket = b;
                count_above = cumulative - histogram[b];
                break;
            }
        }
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint coarse_bucket = threshold_bucket;
    uint already_above = count_above;

    // ============================================================
    // Phase 1b: Fine histogram (bottom 8 bits within coarse bucket)
    // Read from cached sortable buffer (consistent data)
    // ============================================================
    if (lid < NUM_BUCKETS) {
        histogram[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = lid; i < VALUES_NUM; i += WG_SIZE) {
        uint sortable = my_sortable[i];
        if ((sortable >> 8) == coarse_bucket) {
            atomic_add(&histogram[sortable & 0xFF], 1);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find exact threshold
    if (lid == 0) {
        uint cumulative = already_above;
#ifdef MAX_OUT
        for (int b = NUM_BUCKETS - 1; b >= 0; b--) {
            cumulative += histogram[b];
            if (cumulative >= TOP_K) {
                fine_threshold = (coarse_bucket << 8) | (uint)b;
                total_above_threshold = cumulative - histogram[b];
                break;
            }
        }
#else
        for (uint b = 0; b < NUM_BUCKETS; b++) {
            cumulative += histogram[b];
            if (cumulative >= TOP_K) {
                fine_threshold = (coarse_bucket << 8) | b;
                total_above_threshold = cumulative - histogram[b];
                break;
            }
        }
#endif
        gather_count = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint threshold = fine_threshold;

    // ============================================================
    // Initialize SLM sort buffers with fill values
    // Combined key = (sortable << 16) | tiebreaker
    // For MAX_OUT (descending): fill=0 so unused slots sort to end
    // For MIN_OUT (ascending): fill=0xFFFFFFFF so unused slots sort to end
    // ============================================================
    for (uint i = lid; i < PADDED_K; i += WG_SIZE) {
#ifdef MAX_OUT
        sort_keys[i] = 0;
#else
        sort_keys[i] = 0xFFFFFFFF;
#endif
        sort_idxs[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ============================================================
    // Phase 2a: Gather elements strictly ABOVE threshold into SLM
    // Use combined key: (sortable << 16) | tiebreaker(index) for deterministic ordering
    // Read from cached sortable buffer
    // ============================================================
    for (uint i = lid; i < VALUES_NUM; i += WG_SIZE) {
        uint sortable = my_sortable[i];
        bool is_above;
#ifdef MAX_OUT
        is_above = (sortable > threshold);
#else
        is_above = (sortable < threshold);
#endif
        if (is_above) {
            uint pos = atomic_add(&gather_count, 1);
            if (pos < PADDED_K) {
                // Combined key encodes value and index for deterministic tiebreak
#ifdef MAX_OUT
                sort_keys[pos] = (sortable << 16) | (0xFFFF - (i & 0xFFFF));
#else
                sort_keys[pos] = (sortable << 16) | (i & 0xFFFF);
#endif
                sort_idxs[pos] = i;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ============================================================
    // Phase 2b: Gather elements AT threshold into SLM
    // Strategy: check if all at-threshold elements fit in remaining slots.
    //   - Fast path (common): all fit. parallel atomic_add (no correctness issue)
    //   - Slow path (rare):   overflow. single WI sequential scan for deterministic
    //     index ordering (smallest indices first, matching TF/ONNX behavior)
    // histogram[threshold & 0xFF] still holds the fine histogram count from Phase 1.
    // ============================================================
    uint current_count = min(gather_count, (uint)PADDED_K);
    if (lid == 0) {
        gather_count = current_count;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (current_count < PADDED_K) {
        uint at_count = histogram[threshold & 0xFF];
        uint available_slots = PADDED_K - current_count;

        if (at_count <= available_slots) {
            // Fast path: all at-threshold elements fit, parallel gather is safe
            for (uint i = lid; i < VALUES_NUM; i += WG_SIZE) {
                uint sortable = my_sortable[i];
                if (sortable == threshold) {
                    uint pos = atomic_add(&gather_count, 1);
                    if (pos < PADDED_K) {
#ifdef MAX_OUT
                        sort_keys[pos] = (sortable << 16) | (0xFFFF - (i & 0xFFFF));
#else
                        sort_keys[pos] = (sortable << 16) | (i & 0xFFFF);
#endif
                        sort_idxs[pos] = i;
                    }
                }
            }
        } else {
            // Slow path: more at-threshold elements than slots,
            // sequential scan ensures smallest indices are selected first
            if (lid == 0) {
                uint pos = current_count;
                for (uint i = 0; i < VALUES_NUM && pos < PADDED_K; i++) {
                    uint sortable = my_sortable[i];
                    if (sortable == threshold) {
#ifdef MAX_OUT
                        sort_keys[pos] = (sortable << 16) | (0xFFFF - (i & 0xFFFF));
#else
                        sort_keys[pos] = (sortable << 16) | (i & 0xFFFF);
#endif
                        sort_idxs[pos] = i;
                        pos++;
                    }
                }
                gather_count = pos;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ============================================================
    // Phase 3: Bitonic sort in SLM
    // For PADDED_K=512: 9*(9+1)/2 = 45 barrier passes
    // ============================================================
    for (uint bk = 2; bk <= PADDED_K; bk <<= 1) {
        for (uint bj = bk >> 1; bj > 0; bj >>= 1) {
            for (uint i = lid; i < PADDED_K; i += WG_SIZE) {
                uint partner = i ^ bj;
                if (partner > i) {
                    uint ki = sort_keys[i];
                    uint kp = sort_keys[partner];
                    bool ascending = ((i & bk) == 0);
#ifdef MAX_OUT
                    ascending = !ascending;
#endif
                    bool need_swap = ascending ? (ki > kp) : (ki < kp);
                    if (need_swap) {
                        sort_keys[i] = kp;
                        sort_keys[partner] = ki;
                        uint ti = sort_idxs[i];
                        sort_idxs[i] = sort_idxs[partner];
                        sort_idxs[partner] = ti;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // ============================================================
    // Phase 4: Write sorted top-K results to output
    // sort_keys contains combined key: (sortable << 16) | tiebreaker
    // Extract upper 16 bits to recover the sortable value
    // ============================================================
    for (uint k = lid; k < TOP_K; k += WG_SIZE) {
        INPUT0_TYPE val = FUNC_CALL(sortable_to_f16)(sort_keys[k] >> 16);
        uint idx = sort_idxs[k];

        base_indices[AXIS] = k;
        uint out_offset = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR
            base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4]);

#ifdef TOP_K_ORDER
        output[out_offset] = TO_OUTPUT_TYPE(val);
#else
        output[out_offset] = TO_OUTPUT_TYPE(idx);
#endif
#ifdef OUTPUT1_TYPE
    #ifdef TOP_K_ORDER
        second_output[out_offset] = TO_OUTPUT1_TYPE(idx);
    #else
        second_output[out_offset] = TO_OUTPUT1_TYPE(val);
    #endif
#endif
    }
}

#undef COMPARE_SIGN
#undef INPUT0_FILL_VAL
#undef AXIS
#undef VALUES_NUM
#undef WG_SIZE
#undef PADDED_K
#undef NUM_BUCKETS
