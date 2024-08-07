// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(pa_kv_cache_update)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* key_data,
    __global const INPUT1_TYPE* value_data,
    __global const INPUT2_TYPE* subsequence_begins,
    __global const INPUT3_TYPE* block_indices,
    __global const INPUT4_TYPE* past_lens,
    __global const INPUT5_TYPE* block_indices_begins,
    __global OUTPUT_TYPE* key_cache_data,
    __global OUTPUT1_TYPE* value_cache_data,
    const __global int* blocked_indexes_start,
    const __global int* blocked_indexes_end,
    const __global int* gws_seq_indexes_correspondence
    )

    // key_cache layout:   [blocks, head_nums, head_size, vllm_block_size]
    // value_cache layout: [blocks, head_nums, vllm_block_size, head_size]
{
    // If the the number of new tokens equals to the number of past_lens elements,
    // then it's the 2nd+ iteration
    if (INPUT0_BATCH_NUM == INPUT4_BATCH_NUM) {
        // 2nd+ token
        const uint seq_idx = (uint)get_global_id(0);
        const uint head_idx = (uint)get_global_id(1);
        const uint sglid = (uint)get_global_id(2);

        const uint seq_len = past_lens[seq_idx];
        const uint current_token_pos_in_block = seq_len % VLLM_BLOCK_SIZE;
        const uint seq_last_block_idx = block_indices_begins[seq_idx + 1] - 1;
        const uint block_idx = block_indices[seq_last_block_idx];

        uint key_value_in_offset = seq_idx * NUM_HEADS * HEAD_SIZE + head_idx * HEAD_SIZE;

        uint key_out_offset = block_idx * NUM_HEADS * HEAD_SIZE * VLLM_BLOCK_SIZE + head_idx * HEAD_SIZE * VLLM_BLOCK_SIZE + current_token_pos_in_block;

        uint value_out_offset = block_idx * NUM_HEADS * HEAD_SIZE * VLLM_BLOCK_SIZE + head_idx * HEAD_SIZE * VLLM_BLOCK_SIZE + current_token_pos_in_block * HEAD_SIZE;

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        //     printf("Update kv_cache (2nd+): %d %d %d: block_idx=%d, seq_len=%d, current_token_pos_in_block=%d, seq_last_block_idx=%d\n",
        //             seq_idx, head_idx, sglid, block_idx, seq_len, current_token_pos_in_block, seq_last_block_idx);
        // }

        #define READ_BLOCK_SIZE 1
        for (uint head_idx_index = 0; head_idx_index < HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
            #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
            #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

            DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

            unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                // printf("Update kv_cache: %d %d %d, key (head_idx_index=%d): %d -> %d. key_value_in_offset=%d, block_idx=%d, seq_len=%d. in_block_offset=%d\n",
                //     seq_idx, head_idx, sglid, head_idx_index, key_value_in_offset + head_idx_index, key_offset, key_value_in_offset, block_idx, seq_len, head_idx * HEAD_SIZE * VLLM_BLOCK_SIZE + current_token_pos_in_block + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE);
                key_cache_data[key_offset] = input_data;
            }

            // if (seq_len == 15 && head_idx == 0) {
            //     printf("%d. %f\n", head_idx_index + sglid, input_data);
            // }

            input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

            unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                // printf("Update kv_cache: %d %d %d, value (head_idx_index=%d): %d -> %d. key_value_in_offset=%d, block_idx=%d, seq_len=%d. in_block_offset=%d\n",
                    // seq_idx, head_idx, sglid, head_idx_index, key_value_in_offset + head_idx_index, value_offset, key_value_in_offset, block_idx, seq_len, head_idx * HEAD_SIZE * VLLM_BLOCK_SIZE + current_token_pos_in_block * HEAD_SIZE + head_idx_index + sglid + SUBGROUP_SIZE * i);
                value_cache_data[value_offset] = input_data;
            }
        }
    } else {
        // 1st token
        const uint block_idx = get_global_id(0);
        const uint head_idx = get_global_id(1);
        const uint sglid = get_global_id(2);

        const uint block_start_pos = blocked_indexes_start[block_idx];
        const uint block_end_pos = blocked_indexes_end[block_idx];
        const uint tokens_num = block_end_pos - block_start_pos;

        uint key_value_in_offset = block_start_pos * NUM_HEADS * HEAD_SIZE +
                                   head_idx * HEAD_SIZE;

        uint key_out_offset = block_indices[block_idx] * NUM_HEADS * HEAD_SIZE * VLLM_BLOCK_SIZE +
                              head_idx * HEAD_SIZE * VLLM_BLOCK_SIZE;

        uint value_out_offset = key_out_offset;

        // if (get_global_id(1) == 0 && get_global_id(2) == 0) {
        //     printf("%d. Update kv_cache (1st): %d %d %d: block_start_pos=%d block_end_pos=%d tokens_num=%d\n",
        //             get_global_id(0), block_idx, head_idx, sglid, block_start_pos, block_end_pos, tokens_num);
        // }

        // TODO: enable optimization
        if (tokens_num == VLLM_BLOCK_SIZE && false) {
            unroll_for (uint token_num = 0; token_num < VLLM_BLOCK_SIZE; token_num++) {
                uint head_idx_index = 0;
                #define READ_BLOCK_SIZE 8
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 4
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 2
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 1
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data;
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data;
                    }
                }

                key_value_in_offset += NUM_HEADS * HEAD_SIZE;
                key_out_offset += 1;
                value_out_offset += HEAD_SIZE;
            }
        } else {
            for (uint i = 0; i < tokens_num; i++) {
                uint head_idx_index = 0;

#ifdef ENABLE_THIS
                #define READ_BLOCK_SIZE 8
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 4
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 2
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }
#endif

                #define READ_BLOCK_SIZE 1
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * VLLM_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data;
                    }

                    input_data = BLOCK_READ(value_data, key_value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data;
                    }
                }

                key_value_in_offset += NUM_HEADS * HEAD_SIZE;
                key_out_offset += 1;
                value_out_offset += HEAD_SIZE;
            }
        }

    }
}
