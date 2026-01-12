// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Constants
#define EPSILON_FOR_NORM 1e-12f
#define SIZEOF_HALF 2
#define COMPRESSED_EXTRA_DIMS 4  // BY_CHANNEL/BY_TOKEN add 4 dims for scale/zp

// Helper macros for computing key cache offset based on compression mode
// All offsets are in BYTES. Data is int8 (1 byte), scale/zp are half (2 bytes each).
#if KV_CACHE_COMPRESSED
    #if KEY_CACHE_QUANT_MODE == 1  // BY_CHANNEL mode
        // Layout: [num_blocks, KV_HEADS_NUM, K_HEAD_SIZE, PAGED_ATTENTION_BLOCK_SIZE+4]
        // Quantization per dimension: each dimension [d] has its own scale/zp
        // Memory layout per dimension:
        //   [token 0..BLOCK_SIZE-1] (int8 data, BLOCK_SIZE bytes)
        //   [scale] (half, 2 bytes) at offset BLOCK_SIZE
        //   [zp] (half, 2 bytes) at offset BLOCK_SIZE+2
        #define KEY_CACHE_OFFSET(physical_block, token_offset, d) \
            ((physical_block) * KV_HEADS_NUM * K_HEAD_SIZE * \
             (PAGED_ATTENTION_BLOCK_SIZE + COMPRESSED_EXTRA_DIMS) + \
             head_idx * K_HEAD_SIZE * (PAGED_ATTENTION_BLOCK_SIZE + COMPRESSED_EXTRA_DIMS) + \
             (d) * (PAGED_ATTENTION_BLOCK_SIZE + COMPRESSED_EXTRA_DIMS) + (token_offset))
        #define KEY_CACHE_SCALE_OFFSET(physical_block, d) \
            ((physical_block) * KV_HEADS_NUM * K_HEAD_SIZE * \
             (PAGED_ATTENTION_BLOCK_SIZE + COMPRESSED_EXTRA_DIMS) + \
             head_idx * K_HEAD_SIZE * (PAGED_ATTENTION_BLOCK_SIZE + COMPRESSED_EXTRA_DIMS) + \
             (d) * (PAGED_ATTENTION_BLOCK_SIZE + COMPRESSED_EXTRA_DIMS) + \
             PAGED_ATTENTION_BLOCK_SIZE)
        #define KEY_CACHE_ZP_OFFSET(physical_block, d) \
            (KEY_CACHE_SCALE_OFFSET(physical_block, d) + SIZEOF_HALF)
    #elif KEY_CACHE_QUANT_MODE == 2  // BY_TOKEN mode
        // Layout: [num_blocks, KV_HEADS_NUM, K_HEAD_SIZE+4, PAGED_ATTENTION_BLOCK_SIZE]
        // Quantization per token: each token shares same scale/zp across all dimensions
        // Memory layout:
        //   Data rows [dim 0..K_HEAD_SIZE-1]: each row has BLOCK_SIZE int8 values
        //   Scale row [dim K_HEAD_SIZE]: BLOCK_SIZE half values (BLOCK_SIZE * 2 bytes)
        //   Padding row [dim K_HEAD_SIZE+1]: unused
        //   ZP row [dim K_HEAD_SIZE+2]: BLOCK_SIZE half values (BLOCK_SIZE * 2 bytes)
        //   Padding row [dim K_HEAD_SIZE+3]: unused
        // NOTE: token_offset is multiplied by SIZEOF_HALF because scale/zp are half (2 bytes)
        //       token 0 scale at byte offset 0-1, token 1 scale at byte offset 2-3, etc.
        #define KEY_CACHE_OFFSET(physical_block, token_offset, d) \
            ((physical_block) * KV_HEADS_NUM * (K_HEAD_SIZE + COMPRESSED_EXTRA_DIMS) * \
             PAGED_ATTENTION_BLOCK_SIZE + \
             head_idx * (K_HEAD_SIZE + COMPRESSED_EXTRA_DIMS) * PAGED_ATTENTION_BLOCK_SIZE + \
             (d) * PAGED_ATTENTION_BLOCK_SIZE + (token_offset))
        #define KEY_CACHE_SCALE_OFFSET(physical_block, token_offset) \
            ((physical_block) * KV_HEADS_NUM * (K_HEAD_SIZE + COMPRESSED_EXTRA_DIMS) * \
             PAGED_ATTENTION_BLOCK_SIZE + \
             head_idx * (K_HEAD_SIZE + COMPRESSED_EXTRA_DIMS) * PAGED_ATTENTION_BLOCK_SIZE + \
             K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + (token_offset) * SIZEOF_HALF)
        #define KEY_CACHE_ZP_OFFSET(physical_block, token_offset) \
            (KEY_CACHE_SCALE_OFFSET(physical_block, token_offset) + SIZEOF_HALF * PAGED_ATTENTION_BLOCK_SIZE)
    #endif
#else
    // Uncompressed layout: [num_blocks, KV_HEADS_NUM, K_HEAD_SIZE, PAGED_ATTENTION_BLOCK_SIZE]
    #define KEY_CACHE_OFFSET(physical_block, token_offset, d) \
        ((physical_block) * KV_HEADS_NUM * K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + \
         head_idx * K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + \
         (d) * PAGED_ATTENTION_BLOCK_SIZE + (token_offset))
#endif

// Helper macro for normalizing a buffer in-place
#define NORMALIZE_BUFFER(buffer, size) \
    do { \
        barrier(CLK_LOCAL_MEM_FENCE); \
        ACCUMULATOR_TYPE norm_sq_temp = 0.0f; \
        for (uint d = sglid; d < size; d += SUBGROUP_SIZE) { \
            norm_sq_temp += buffer[d] * buffer[d]; \
        } \
        norm_sq_temp = sub_group_reduce_add(norm_sq_temp); \
        ACCUMULATOR_TYPE norm_temp = native_sqrt(norm_sq_temp + EPSILON_FOR_NORM); \
        for (uint d = sglid; d < size; d += SUBGROUP_SIZE) { \
            buffer[d] /= norm_temp; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } while(0)

#if KV_CACHE_COMPRESSED
    // Helper macro for loading scale and zero-point for BY_CHANNEL mode
    #define LOAD_SCALE_ZP_BY_CHANNEL(physical_block, d, scale_var, zp_var) \
        do { \
            __global const half* scale_ptr = \
                (__global const half*)&key_cache[KEY_CACHE_SCALE_OFFSET(physical_block, d)]; \
            __global const half* zp_ptr = \
                (__global const half*)&key_cache[KEY_CACHE_ZP_OFFSET(physical_block, d)]; \
            scale_var = (ACCUMULATOR_TYPE)(*scale_ptr); \
            zp_var = (ACCUMULATOR_TYPE)(*zp_ptr); \
        } while(0)

    // Helper macro for loading scale and zero-point for BY_TOKEN mode
    #define LOAD_SCALE_ZP_BY_TOKEN(physical_block, token_offset, scale_var, zp_var) \
        do { \
            __global const half* scale_ptr = \
                (__global const half*)&key_cache[KEY_CACHE_SCALE_OFFSET(physical_block, token_offset)]; \
            __global const half* zp_ptr = \
                (__global const half*)&key_cache[KEY_CACHE_ZP_OFFSET(physical_block, token_offset)]; \
            scale_var = (ACCUMULATOR_TYPE)(*scale_ptr); \
            zp_var = (ACCUMULATOR_TYPE)(*zp_ptr); \
        } while(0)
#endif

// Helper macro for loading and normalizing key vectors
// Separate implementations for compressed vs uncompressed to avoid dummy macros
#if KV_CACHE_COMPRESSED
#define LOAD_AND_NORMALIZE_KEY(buffer, physical_block, token_offset) \
    do { \
        for (int d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) { \
            INPUT0_TYPE raw_value = key_cache[KEY_CACHE_OFFSET(physical_block, token_offset, d)]; \
            ACCUMULATOR_TYPE value; \
            if (KEY_CACHE_QUANT_MODE == 1) { \
                /* BY_CHANNEL: Each dimension has its own scale/zp */ \
                ACCUMULATOR_TYPE scale, zp; \
                LOAD_SCALE_ZP_BY_CHANNEL(physical_block, d, scale, zp); \
                value = ((ACCUMULATOR_TYPE)raw_value - zp) * scale; \
            } else if (KEY_CACHE_QUANT_MODE == 2) { \
                /* BY_TOKEN: All dimensions of a token share the same scale/zp */ \
                ACCUMULATOR_TYPE scale, zp; \
                LOAD_SCALE_ZP_BY_TOKEN(physical_block, token_offset, scale, zp); \
                value = ((ACCUMULATOR_TYPE)raw_value - zp) * scale; \
            } \
            buffer[d] = value; \
        } \
        NORMALIZE_BUFFER(buffer, K_HEAD_SIZE); \
    } while(0)
#else
// Uncompressed version - no dequantization needed
#define LOAD_AND_NORMALIZE_KEY(buffer, physical_block, token_offset) \
    do { \
        for (int d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) { \
            buffer[d] = (ACCUMULATOR_TYPE)key_cache[KEY_CACHE_OFFSET(physical_block, token_offset, d)]; \
        } \
        NORMALIZE_BUFFER(buffer, K_HEAD_SIZE); \
    } while(0)
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1)))
KERNEL(pa_adaptive_rkv_diversity)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* key_cache,              // [num_blocks, KV_HEADS_NUM, K_HEAD_SIZE, BLOCK_SIZE]
    __global const INPUT1_TYPE* evictable_sizes,        // [batch_size]
    __global const INPUT2_TYPE* block_indices,          // [total_evictable_blocks]
    __global const INPUT3_TYPE* block_indices_begins,   // [batch_size + 1]
    __global OUTPUT_TYPE* diversity_output,             // [total_diversity_scores]
#ifdef USE_GLOBAL_BUFFERS
    __global ACCUMULATOR_TYPE* similarity_matrix,       // global buffer for similarity matrix
    __global ACCUMULATOR_TYPE* aggregated_similarities, // global buffer for aggregated similarities
    __global ACCUMULATOR_TYPE* row_means,               // global buffer for row means
    __global ACCUMULATOR_TYPE* block_sums,              // global buffer for block sums
#endif
    const int start_size                                // scalar
) {
    const uint batch_idx = get_group_id(0);
    const uint sglid = get_sub_group_local_id();
    const int evictable_size = evictable_sizes[batch_idx];
    const int num_evictable_blocks = evictable_size / PAGED_ATTENTION_BLOCK_SIZE;

    if (num_evictable_blocks == 0)
        return;

    // Calculate diversity output offset for this batch
    // Each batch contributes (evictable_size / block_size) * evictable_size elements
    int diversity_output_offset = 0;
    for (int b = 0; b < batch_idx; b++) {
        int prev_evictable_size = evictable_sizes[b];
        int prev_num_blocks = prev_evictable_size / PAGED_ATTENTION_BLOCK_SIZE;
        diversity_output_offset += prev_num_blocks * prev_evictable_size;
    }

    // Get block indices range for this batch
    const int block_begin_idx = block_indices_begins[batch_idx];
    const int block_end_idx = block_indices_begins[batch_idx + 1];
    const int num_evictable_blocks_for_batch = block_end_idx - block_begin_idx;

    if (num_evictable_blocks_for_batch != num_evictable_blocks)
        return;

#ifdef USE_GLOBAL_BUFFERS
    // Use fixed-size allocation per batch for simplicity and efficiency
    // Each batch gets MAX_EVICTABLE_SIZE^2 space regardless of actual size
    const int max_matrix_size = MAX_EVICTABLE_SIZE * MAX_EVICTABLE_SIZE;

    // Calculate offsets using fixed batch stride
    int batch_matrix_offset = batch_idx * max_matrix_size;
    int batch_vector_offset = batch_idx * MAX_EVICTABLE_SIZE;

    // Offset pointers to this batch's region
    __global ACCUMULATOR_TYPE* similarity_matrix_batch = similarity_matrix + batch_matrix_offset;
    __global ACCUMULATOR_TYPE* aggregated_similarities_batch = aggregated_similarities + batch_matrix_offset;
    __global ACCUMULATOR_TYPE* row_means_batch = row_means + batch_vector_offset;
    __global ACCUMULATOR_TYPE* block_sums_batch = block_sums + batch_vector_offset;
#else
    // Local memory fallback (limited to ~64KB SLM)
    __local ACCUMULATOR_TYPE similarity_matrix_batch[4096];           // 64×64 matrix (max for local memory)
    __local ACCUMULATOR_TYPE aggregated_similarities_batch[4096];     // 64×64 aggregated matrix
    __local ACCUMULATOR_TYPE row_means_batch[256];                    // Row means vector (max 256 tokens)
    __local ACCUMULATOR_TYPE block_sums_batch[256];                   // Block sum vector (max 256 tokens)
#endif

    // Keep normalized_key_i in local memory (small, frequently reused)
    __local ACCUMULATOR_TYPE normalized_key_i[K_HEAD_SIZE];

    // Set appropriate memory fence based on buffer type
#ifdef USE_GLOBAL_BUFFERS
    const cl_mem_fence_flags mem_fence = CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE;
#else
    const cl_mem_fence_flags mem_fence = CLK_LOCAL_MEM_FENCE;
#endif

    // Initialize aggregated similarities (full matrix)
    for (int i = sglid; i < evictable_size * evictable_size; i += SUBGROUP_SIZE) {
        aggregated_similarities_batch[i] = 0.0f;
    }
    barrier(mem_fence);
    for (uint head_idx = 0; head_idx < KV_HEADS_NUM; head_idx++) {
        // Initialize similarity matrix for this head
        for (int i = sglid; i < evictable_size * evictable_size; i += SUBGROUP_SIZE) {
            similarity_matrix_batch[i] = 0.0f;
        }
        barrier(mem_fence);

        // Compute cosine similarity matrix for evictable region
        // Exploit symmetry: only compute upper triangle (token_j >= token_i), then mirror
        for (int token_i = 0; token_i < evictable_size; token_i++) {
            const int abs_token_i = start_size + token_i;
            const int block_idx_i = abs_token_i / PAGED_ATTENTION_BLOCK_SIZE;
            const int token_offset_i = abs_token_i % PAGED_ATTENTION_BLOCK_SIZE;
            const int evictable_block_idx_i = block_idx_i - start_size / PAGED_ATTENTION_BLOCK_SIZE;
            const int physical_block_i = block_indices[block_begin_idx + evictable_block_idx_i];

            // Load and normalize key for token_i (reused across all token_j in this row)
            LOAD_AND_NORMALIZE_KEY(normalized_key_i, physical_block_i, token_offset_i);

            // Compute similarities only with token_j >= token_i (upper triangle)
            for (int token_j = token_i; token_j < evictable_size; token_j++) {
                const int abs_token_j = start_size + token_j;
                const int block_idx_j = abs_token_j / PAGED_ATTENTION_BLOCK_SIZE;
                const int token_offset_j = abs_token_j % PAGED_ATTENTION_BLOCK_SIZE;
                const int evictable_block_idx_j = block_idx_j - start_size / PAGED_ATTENTION_BLOCK_SIZE;
                const int physical_block_j = block_indices[block_begin_idx + evictable_block_idx_j];

                ACCUMULATOR_TYPE similarity;
                if (token_i == token_j) {
                    similarity = 1.0f;
                } else {
                    // Load token_j on-the-fly (not buffered) and compute cosine similarity
                    // Formula: cosine(i,j) = dot(normalized_i, j) / norm(j)
                    ACCUMULATOR_TYPE dot_product = 0.0f;
                    ACCUMULATOR_TYPE norm_sq_j = 0.0f;

#if KV_CACHE_COMPRESSED
                    // Compressed: dequantize during loading
                    for (int d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) {
                        INPUT0_TYPE raw_value = key_cache[KEY_CACHE_OFFSET(physical_block_j, token_offset_j, d)];
                        ACCUMULATOR_TYPE key_j_val;
                        if (KEY_CACHE_QUANT_MODE == 1) {
                            // BY_CHANNEL: Each dimension [d] has its own scale/zp
                            ACCUMULATOR_TYPE scale, zp;
                            LOAD_SCALE_ZP_BY_CHANNEL(physical_block_j, d, scale, zp);
                            key_j_val = ((ACCUMULATOR_TYPE)raw_value - zp) * scale;
                        } else if (KEY_CACHE_QUANT_MODE == 2) {
                            // BY_TOKEN: All dimensions of token_j share the same scale/zp
                            ACCUMULATOR_TYPE scale, zp;
                            LOAD_SCALE_ZP_BY_TOKEN(physical_block_j, token_offset_j, scale, zp);
                            key_j_val = ((ACCUMULATOR_TYPE)raw_value - zp) * scale;
                        }
                        dot_product += normalized_key_i[d] * key_j_val;
                        norm_sq_j += key_j_val * key_j_val;
                    }
#else
                    // Uncompressed: direct loading
                    for (int d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) {
                        ACCUMULATOR_TYPE key_j_val = (ACCUMULATOR_TYPE)key_cache[KEY_CACHE_OFFSET(physical_block_j, token_offset_j, d)];
                        dot_product += normalized_key_i[d] * key_j_val;
                        norm_sq_j += key_j_val * key_j_val;
                    }
#endif

                    // Subgroup reductions
                    dot_product = sub_group_reduce_add(dot_product);
                    norm_sq_j = sub_group_reduce_add(norm_sq_j);
                    ACCUMULATOR_TYPE norm_j = native_sqrt(norm_sq_j + EPSILON_FOR_NORM);

                    similarity = dot_product / norm_j;
                }

                // Write to both upper and lower triangle (exploiting symmetry)
                if (sglid == 0) {
                    similarity_matrix_batch[token_i * evictable_size + token_j] = similarity;
                    similarity_matrix_batch[token_j * evictable_size + token_i] = similarity;
                }
            }
        }

        // Fill diagonal with 0 for this head
        for (int i = sglid; i < evictable_size; i += SUBGROUP_SIZE) {
            similarity_matrix_batch[i * evictable_size + i] = 0.0f;
        }
        barrier(mem_fence);

        // Compute row-wise mean for this head
        for (int row = sglid; row < evictable_size; row += SUBGROUP_SIZE) {
            ACCUMULATOR_TYPE row_sum = 0.0f;
            for (int col = 0; col < evictable_size; col++) {
                row_sum += similarity_matrix_batch[row * evictable_size + col];
            }
            row_means_batch[row] = row_sum / (ACCUMULATOR_TYPE)evictable_size;
        }
        barrier(mem_fence);

        // Apply threshold for this head: set values below mean to 0
        for (int idx = sglid; idx < evictable_size * evictable_size; idx += SUBGROUP_SIZE) {
            int row = idx / evictable_size;
            ACCUMULATOR_TYPE val = similarity_matrix_batch[idx];
            if (val < row_means_batch[row]) {
                similarity_matrix_batch[idx] = 0.0f;
            }
        }
        barrier(mem_fence);

        // Accumulate thresholded values across heads
        for (int idx = sglid; idx < evictable_size * evictable_size; idx += SUBGROUP_SIZE) {
            aggregated_similarities_batch[idx] += similarity_matrix_batch[idx];
        }
        barrier(mem_fence);
    }

    // Reduce mean across heads
    for (int idx = sglid; idx < evictable_size * evictable_size; idx += SUBGROUP_SIZE) {
        aggregated_similarities_batch[idx] /= (ACCUMULATOR_TYPE)KV_HEADS_NUM;
    }
    barrier(mem_fence);

    // Block sum (negative sum): sum block_size rows for each block
    // Output: flattened 2D [num_evictable_blocks, evictable_size]
    for (int block_idx = 0; block_idx < num_evictable_blocks; block_idx++) {
        // Initialize block sums for this block (all work items participate)
        for (int col = sglid; col < evictable_size; col += SUBGROUP_SIZE) {
            ACCUMULATOR_TYPE sum = 0.0f;
            int row_start = block_idx * PAGED_ATTENTION_BLOCK_SIZE;
            for (int row_offset = 0; row_offset < PAGED_ATTENTION_BLOCK_SIZE; row_offset++) {
                int row = row_start + row_offset;
                if (row < evictable_size) {
                    // Negative sum (as in reference)
                    sum -= aggregated_similarities_batch[row * evictable_size + col];
                }
            }
            block_sums_batch[col] = sum;
        }
        barrier(mem_fence);

        // Write all block_sums directly to output (flattened 2D: [num_evictable_blocks, evictable_size])
        for (int col = sglid; col < evictable_size; col += SUBGROUP_SIZE) {
            const int output_offset = diversity_output_offset + block_idx * evictable_size + col;
            diversity_output[output_offset] = (OUTPUT_TYPE)block_sums_batch[col];
        }
        barrier(mem_fence);
    }
}
