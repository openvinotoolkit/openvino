// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Helper macro for computing key cache offset
#define KEY_CACHE_OFFSET(physical_block, token_offset, d) \
    ((physical_block) * KV_HEADS_NUM * K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + \
     head_idx * K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + \
     (d) * PAGED_ATTENTION_BLOCK_SIZE + (token_offset))

// Helper macro for loading and normalizing key vectors
#define LOAD_AND_NORMALIZE_KEY(buffer, physical_block, token_offset) \
    do { \
        for (int d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) { \
            buffer[d] = (ACCUMULATOR_TYPE)key_cache[KEY_CACHE_OFFSET(physical_block, token_offset, d)]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        ACCUMULATOR_TYPE norm_sq_temp = 0.0f; \
        for (uint d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) { \
            norm_sq_temp += buffer[d] * buffer[d]; \
        } \
        norm_sq_temp = sub_group_reduce_add(norm_sq_temp); \
        ACCUMULATOR_TYPE norm_temp = native_sqrt(norm_sq_temp + 1e-12f); \
        for (uint d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) { \
            buffer[d] /= norm_temp; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } while(0)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1)))
KERNEL(pa_adaptive_rkv_diversity)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* key_cache,              // [num_blocks, KV_HEADS_NUM, K_HEAD_SIZE, BLOCK_SIZE]
    __global const INPUT1_TYPE* start_sizes,            // [batch_size]
    __global const INPUT2_TYPE* evictable_sizes,        // [batch_size]
    __global const INPUT3_TYPE* block_indices,          // [total_evictable_blocks]
    __global const INPUT4_TYPE* block_indices_begins,   // [batch_size + 1]
    __global OUTPUT_TYPE* diversity_output              // [total_diversity_scores]
) {
    const uint batch_idx = get_group_id(0);
    const uint sglid = get_sub_group_local_id();

    const int start_size = start_sizes[batch_idx];
    const int evictable_size = evictable_sizes[batch_idx];
    const int num_evictable_blocks = evictable_size / PAGED_ATTENTION_BLOCK_SIZE;

    if (sglid == 0 && batch_idx == 0) {
        printf("[DEBUG] batch_idx=%d, start_size=%d, evictable_size=%d, num_evictable_blocks=%d\n",
               batch_idx, start_size, evictable_size, num_evictable_blocks);
    }

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

    // Allocate local memory
    __local ACCUMULATOR_TYPE normalized_key_i[K_HEAD_SIZE];
    __local ACCUMULATOR_TYPE similarity_matrix[4096];           // 64*64 max evictable_size
    __local ACCUMULATOR_TYPE aggregated_similarities[4096];     // After head aggregation
    __local ACCUMULATOR_TYPE row_means[256];                    // Max evictable_size
    __local ACCUMULATOR_TYPE block_sums[256];                   // Max evictable_size for intermediate storage

    // Initialize aggregated similarities (full matrix)
    for (int i = sglid; i < evictable_size * evictable_size; i += SUBGROUP_SIZE) {
        aggregated_similarities[i] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Process each head - threshold BEFORE aggregation
    for (uint head_idx = 0; head_idx < KV_HEADS_NUM; head_idx++) {
        // Initialize similarity matrix for this head
        for (int i = sglid; i < evictable_size * evictable_size; i += SUBGROUP_SIZE) {
            similarity_matrix[i] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

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

                    for (int d = sglid; d < K_HEAD_SIZE; d += SUBGROUP_SIZE) {
                        ACCUMULATOR_TYPE key_j_val = (ACCUMULATOR_TYPE)key_cache[KEY_CACHE_OFFSET(physical_block_j, token_offset_j, d)];
                        dot_product += normalized_key_i[d] * key_j_val;
                        norm_sq_j += key_j_val * key_j_val;
                    }

                    // Subgroup reductions
                    dot_product = sub_group_reduce_add(dot_product);
                    norm_sq_j = sub_group_reduce_add(norm_sq_j);
                    ACCUMULATOR_TYPE norm_j = native_sqrt(norm_sq_j + 1e-12f);

                    similarity = dot_product / norm_j;
                }

                // Write to both upper and lower triangle (exploiting symmetry)
                if (sglid == 0) {
                    similarity_matrix[token_i * evictable_size + token_j] = similarity;
                    similarity_matrix[token_j * evictable_size + token_i] = similarity;
                }
            }
        }

        // Fill diagonal with 0 for this head
        for (int i = sglid; i < evictable_size; i += SUBGROUP_SIZE) {
            similarity_matrix[i * evictable_size + i] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute row-wise mean for this head
        for (int row = sglid; row < evictable_size; row += SUBGROUP_SIZE) {
            ACCUMULATOR_TYPE row_sum = 0.0f;
            for (int col = 0; col < evictable_size; col++) {
                row_sum += similarity_matrix[row * evictable_size + col];
            }
            row_means[row] = row_sum / (ACCUMULATOR_TYPE)evictable_size;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Apply threshold for this head: set values below mean to 0
        for (int idx = sglid; idx < evictable_size * evictable_size; idx += SUBGROUP_SIZE) {
            int row = idx / evictable_size;
            ACCUMULATOR_TYPE val = similarity_matrix[idx];
            if (val < row_means[row]) {
                similarity_matrix[idx] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate thresholded values across heads
        for (int idx = sglid; idx < evictable_size * evictable_size; idx += SUBGROUP_SIZE) {
            aggregated_similarities[idx] += similarity_matrix[idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Reduce mean across heads
    for (int idx = sglid; idx < evictable_size * evictable_size; idx += SUBGROUP_SIZE) {
        aggregated_similarities[idx] /= (ACCUMULATOR_TYPE)KV_HEADS_NUM;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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
                    sum -= aggregated_similarities[row * evictable_size + col];
                }
            }
            block_sums[col] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Write all block_sums directly to output (flattened 2D: [num_evictable_blocks, evictable_size])
        for (int col = sglid; col < evictable_size; col += SUBGROUP_SIZE) {
            const int output_offset = diversity_output_offset + block_idx * evictable_size + col;
            diversity_output[output_offset] = (OUTPUT_TYPE)block_sums[col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
