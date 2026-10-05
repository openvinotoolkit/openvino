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
// Chunked mode (RADIX_CHUNK_STAGE + RADIX_MERGE_STAGE kernels):
//   One work-group per operation leaves most of the GPU idle when there are only a few
//   long rows. The chunk stage runs the algorithm above on RADIX_CHUNKS slices of each
//   row and stores every slice's top-K keys and indices. The merge stage runs it again
//   on the RADIX_CANDIDATES_NUM survivors of each row and writes the outputs. Survivors
//   are laid out by chunk and sorted with the same (key, index) ordering, so among equal
//   keys their position order matches the original index order.
//
// Design principles:
//   - Read input exactly ONCE to avoid GPU cache inconsistency
//   - Single global buffer for sortable keys (N * 4 bytes per operation)
//   - SLM bitonic sort for Phase 3 (fast, no global memory for sorting)
//   - Total SLM: ~6KB (histogram 1KB + tie counters 1KB + sort buffers 4KB)
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
#define BUCKET_GROUP_SIZE 16
#define BUCKET_GROUPS (NUM_BUCKETS / BUCKET_GROUP_SIZE)
#define TIE_GROUP_SIZE 16
#define TIE_GROUPS (WG_SIZE / TIE_GROUP_SIZE)

// Coarse-to-fine radix select: one 8-bit digit per pass, MSD first.
#define NUM_PASSES (SORTABLE_BITS / 8)

// Ordering of (key, index) pairs. Equal values are broken by the smallest original index,
// matching TF/ONNX TopK behaviour.
#ifdef MAX_OUT
    #define PAIR_GT(key_a, idx_a, key_b, idx_b) ((key_a) != (key_b) ? (key_a) > (key_b) : (idx_a) < (idx_b))
#else
    #define PAIR_GT(key_a, idx_a, key_b, idx_b) ((key_a) != (key_b) ? (key_a) > (key_b) : (idx_a) > (idx_b))
#endif
#define FILL_IDX 0xFFFFFFFFu

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

// Convert a float value to a monotonically increasing unsigned integer:
// for positive values the bit pattern is already ordered, so only the sign bit is flipped;
// for negative values all bits are flipped.
#if SORTABLE_BITS == 16
inline uint FUNC(to_sortable)(INPUT0_TYPE val) {
    ushort bits = as_ushort(convert_half(val));
    ushort mask = (bits >> 15) ? (ushort)0xFFFF : (ushort)0x8000;
    return (uint)(bits ^ mask);
}

inline INPUT0_TYPE FUNC(from_sortable)(uint sortable) {
    ushort bits = (ushort)sortable;
    ushort mask = (bits & 0x8000) ? (ushort)0x8000 : (ushort)0xFFFF;
    return TO_INPUT0_TYPE(as_half((ushort)(bits ^ mask)));
}
#else
inline uint FUNC(to_sortable)(INPUT0_TYPE val) {
    uint bits = as_uint(convert_float(val));
    uint mask = (bits >> 31) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

inline INPUT0_TYPE FUNC(from_sortable)(uint sortable) {
    uint mask = (sortable & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return TO_INPUT0_TYPE(as_float(sortable ^ mask));
}
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(arg_max_min_topk_radix)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input
    ,__global OUTPUT_TYPE* output
#ifdef OUTPUT1_TYPE
    ,__global OUTPUT1_TYPE* second_output
#endif
    ,__global uint* sortable_buf            // Cached sortable keys: VALUES_NUM per operation
#if defined(RADIX_CHUNK_STAGE) || defined(RADIX_MERGE_STAGE)
    ,__global uint* candidate_keys          // Per-chunk top-K keys: RADIX_CANDIDATES_NUM per operation
    ,__global uint* candidate_idxs          // Original indices of candidate_keys
#endif
)
{
    const uint lid = (uint)get_local_id(0);
#ifdef RADIX_CHUNK_STAGE
    const uint chunk_group_idx = (uint)get_group_id(0);
    const uint output_idx = chunk_group_idx / RADIX_CHUNKS;
    const uint chunk_begin = (chunk_group_idx % RADIX_CHUNKS) * RADIX_CHUNK_SIZE;
    const uint values_count = min((uint)RADIX_CHUNK_SIZE, (uint)VALUES_NUM - chunk_begin);
    __global uint* my_sortable = sortable_buf + chunk_group_idx * RADIX_CHUNK_SIZE;
#elif defined(RADIX_MERGE_STAGE)
    const uint output_idx = (uint)get_group_id(0);
    const uint values_count = RADIX_CANDIDATES_NUM;
    // Keys were already converted by the chunk stage; positions index candidate_idxs.
    const __global uint* my_sortable = candidate_keys + output_idx * RADIX_CANDIDATES_NUM;
#else
    const uint output_idx = (uint)get_group_id(0);
    const uint chunk_begin = 0;
    const uint values_count = VALUES_NUM;
    __global uint* my_sortable = sortable_buf + output_idx * VALUES_NUM;
#endif

    uint base_indices[] = { 0, 0, 0, 0, 0 };
    if (OPERATION_NUM > 1) {
        FUNC_CALL(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_TENSOR output_idx, base_indices);
    }

    // ============================================================
    // SLM declarations (all at outermost kernel scope)
    // Total SLM: 256*4 + 16*4 + 256*4 + 16*4 + 512*4 + 512*4 + ~20 ~= 6KB
    // ============================================================
    __local uint histogram[NUM_BUCKETS];        // 1KB
    __local uint bucket_group_counts[BUCKET_GROUPS];
    __local uint tie_counts[WG_SIZE];           // per work-item tie counts for ordered tie gathering
    __local uint tie_group_counts[TIE_GROUPS];
    __local uint sort_keys[PADDED_K];           // sortable uint keys for bitonic sort
    __local uint sort_idxs[PADDED_K];           // original indices
    __local uint threshold_bucket;
    __local uint count_above;
    __local uint gather_count;

#ifndef RADIX_MERGE_STAGE
    // ============================================================
    // Phase 0: Read input ONCE, convert to sortable keys, cache in global buffer
    // ============================================================
    for (uint i = lid; i < values_count; i += WG_SIZE) {
        base_indices[AXIS] = chunk_begin + i;
        INPUT0_TYPE val = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR
            base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4])];
        my_sortable[i] = FUNC_CALL(to_sortable)(val);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif

    // ============================================================
    // Phase 1: MSD radix select - resolve the K-th key one byte at a time.
    // All NUM_PASSES bytes are resolved so the threshold is an exact key value,
    // which is what phase 2b relies on to break ties by index.
    // ============================================================
    uint prefix = 0;         // bytes of the threshold resolved so far
    uint already_above = 0;  // elements strictly beyond the resolved prefix

    for (uint pass = 0; pass < NUM_PASSES; pass++) {
        const uint shift = SORTABLE_BITS - 8 * (pass + 1);
        const uint hi_shift = shift + 8;

        if (lid < NUM_BUCKETS) {
            histogram[lid] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = lid; i < values_count; i += WG_SIZE) {
            uint sortable = my_sortable[i];
            uint hi = (hi_shift >= SORTABLE_BITS) ? 0u : (sortable >> hi_shift);
            if (hi == prefix) {
                atomic_add(&histogram[(sortable >> shift) & 0xFF], 1);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Two-level search for the threshold bucket: group sums first, then buckets of one group.
        if (lid < BUCKET_GROUPS) {
            uint group_count = 0;
            for (uint j = 0; j < BUCKET_GROUP_SIZE; j++) {
                group_count += histogram[lid * BUCKET_GROUP_SIZE + j];
            }
            bucket_group_counts[lid] = group_count;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            uint cumulative = already_above;
#ifdef MAX_OUT
            int group = BUCKET_GROUPS - 1;
            while (group > 0 && cumulative + bucket_group_counts[group] < TOP_K) {
                cumulative += bucket_group_counts[group];
                group--;
            }
            for (int b = (group + 1) * BUCKET_GROUP_SIZE - 1; b >= 0; b--) {
                cumulative += histogram[b];
                if (cumulative >= TOP_K) {
                    threshold_bucket = (uint)b;
                    count_above = cumulative - histogram[b];
                    break;
                }
            }
#else
            uint group = 0;
            while (group < BUCKET_GROUPS - 1 && cumulative + bucket_group_counts[group] < TOP_K) {
                cumulative += bucket_group_counts[group];
                group++;
            }
            for (uint b = group * BUCKET_GROUP_SIZE; b < NUM_BUCKETS; b++) {
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

        prefix = (prefix << 8) | threshold_bucket;
        already_above = count_above;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        gather_count = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint threshold = prefix;

    // ============================================================
    // Initialize SLM sort buffers with fill values so unused slots sort to the end
    // ============================================================
    for (uint i = lid; i < PADDED_K; i += WG_SIZE) {
#ifdef MAX_OUT
        sort_keys[i] = 0u;
#else
        sort_keys[i] = 0xFFFFFFFFu;
#endif
        sort_idxs[i] = FILL_IDX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ============================================================
    // Phase 2a: Gather elements strictly ABOVE threshold into SLM
    // Read from cached sortable buffer
    // ============================================================
    for (uint i = lid; i < values_count; i += WG_SIZE) {
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
                sort_keys[pos] = sortable;
                sort_idxs[pos] = i;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ============================================================
    // Phase 2b: Gather elements AT threshold into SLM
    // Strategy: check if all at-threshold elements fit in remaining slots.
    //   - Fast path: all fit. parallel atomic_add (no correctness issue)
    //   - Ordered path: overflow. Each work-item owns a contiguous index block, so a
    //     prefix sum of per-block tie counts places the smallest indices first
    //     (matching TF/ONNX behavior) without a single work-item scan.
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
            for (uint i = lid; i < values_count; i += WG_SIZE) {
                uint sortable = my_sortable[i];
                if (sortable == threshold) {
                    uint pos = atomic_add(&gather_count, 1);
                    if (pos < PADDED_K) {
                        sort_keys[pos] = sortable;
                        sort_idxs[pos] = i;
                    }
                }
            }
        } else {
            const uint block_size = (values_count + WG_SIZE - 1) / WG_SIZE;
            const uint block_begin = min(lid * block_size, values_count);
            const uint block_end = min(block_begin + block_size, values_count);

            uint block_ties = 0;
            for (uint i = block_begin; i < block_end; i++) {
                block_ties += (my_sortable[i] == threshold) ? 1u : 0u;
            }
            tie_counts[lid] = block_ties;
            barrier(CLK_LOCAL_MEM_FENCE);

            const uint tie_group = lid / TIE_GROUP_SIZE;
            if (lid < TIE_GROUPS) {
                uint group_ties = 0;
                for (uint j = 0; j < TIE_GROUP_SIZE; j++) {
                    group_ties += tie_counts[lid * TIE_GROUP_SIZE + j];
                }
                tie_group_counts[lid] = group_ties;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (block_ties != 0) {
                uint pos = current_count;
                for (uint g = 0; g < tie_group; g++) {
                    pos += tie_group_counts[g];
                }
                for (uint j = tie_group * TIE_GROUP_SIZE; j < lid; j++) {
                    pos += tie_counts[j];
                }

                for (uint i = block_begin; i < block_end && pos < PADDED_K; i++) {
                    if (my_sortable[i] == threshold) {
                        sort_keys[pos] = threshold;
                        sort_idxs[pos] = i;
                        pos++;
                    }
                }
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
                    uint key_i = sort_keys[i];
                    uint key_p = sort_keys[partner];
                    uint idx_i = sort_idxs[i];
                    uint idx_p = sort_idxs[partner];
                    bool ascending = ((i & bk) == 0);
#ifdef MAX_OUT
                    ascending = !ascending;
#endif
                    bool need_swap = ascending ? PAIR_GT(key_i, idx_i, key_p, idx_p)
                                               : PAIR_GT(key_p, idx_p, key_i, idx_i);
                    if (need_swap) {
                        sort_keys[i] = key_p;
                        sort_keys[partner] = key_i;
                        sort_idxs[i] = idx_p;
                        sort_idxs[partner] = idx_i;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#ifdef RADIX_CHUNK_STAGE
    // ============================================================
    // Phase 4: Store this chunk's sorted top-K as merge candidates
    // ============================================================
    for (uint k = lid; k < TOP_K; k += WG_SIZE) {
        candidate_keys[chunk_group_idx * TOP_K + k] = sort_keys[k];
        candidate_idxs[chunk_group_idx * TOP_K + k] = chunk_begin + sort_idxs[k];
    }
#else
    // ============================================================
    // Phase 4: Write sorted top-K results to output
    // ============================================================
    for (uint k = lid; k < TOP_K; k += WG_SIZE) {
        INPUT0_TYPE val = FUNC_CALL(from_sortable)(sort_keys[k]);
#ifdef RADIX_MERGE_STAGE
        uint idx = candidate_idxs[output_idx * RADIX_CANDIDATES_NUM + sort_idxs[k]];
#else
        uint idx = sort_idxs[k];
#endif

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
#endif
}

#undef COMPARE_SIGN
#undef INPUT0_FILL_VAL
#undef AXIS
#undef VALUES_NUM
#undef WG_SIZE
#undef PADDED_K
#undef NUM_BUCKETS
#undef BUCKET_GROUP_SIZE
#undef BUCKET_GROUPS
#undef TIE_GROUP_SIZE
#undef TIE_GROUPS
#undef SORTABLE_BITS
#undef NUM_PASSES
#undef PAIR_GT
#undef FILL_IDX
