// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#define INT4_RANGE 15
#define INT4_MAX 7
#define INT4_MIN (-8)
#define PACK_SIZE 2
#define ENABLE_DEBUG 0


inline void FUNC(quantize_and_save_per_token)(__global const INPUT0_TYPE* in_data,
                                    const uint in_data_offset,
                                    __global OUTPUT_TYPE* out_data,
                                    const uint out_data_offset,
                                    const uint out_data_pitch,
                                    const uint comp_offset,
                                    const uint token_pos_in_block,
                                    const uint sglid,
                                    const uint num_groups,
                                    INPUT0_TYPE* input_data) {
    INPUT0_TYPE grp_max = 0.001;
    INPUT0_TYPE max_value = INPUT0_VAL_MIN;
    INPUT0_TYPE min_value = INPUT0_VAL_MAX;

    unroll_for (uint i = 0; i < num_groups; i++) {
        input_data[i] = BLOCK_READN(INPUT0_TYPE, 1, in_data, in_data_offset + i * SUBGROUP_SIZE);
        max_value = fmax(max_value, input_data[i]);
        min_value = fmin(min_value, input_data[i]);
    }

    min_value = sub_group_reduce_min(min_value);
    max_value = sub_group_reduce_max(max_value);

    // If the range of input data is zero, it is adjusted to the minimum value(0.001).
    #define ACCUMULATOR_TYPE float
    ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);

    #if IS_INT4_COMPRESSED
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((INT4_RANGE) / diff_value);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + INT4_MIN;
    #else
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((INT4_MAX - INT4_MIN) / diff_value);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + INT4_MIN;
    #endif
    INPUT0_TYPE scale = (INPUT1_TYPE)(scale_tmp);
    INPUT0_TYPE zp = (INPUT1_TYPE)(zp_tmp);

    #undef ACCUMULATOR_TYPE

    #if IS_INT4_COMPRESSED
        #if ENABLE_DEBUG
        OUTPUT_TYPE result[(K_HEAD_SIZE+V_HEAD_SIZE) / SUBGROUP_SIZE] = {0, };
        OUTPUT_TYPE orig_result[(K_HEAD_SIZE+V_HEAD_SIZE) / SUBGROUP_SIZE] = {0, };
        {
            if ((uint)get_global_id(0) == 0 && (uint)get_global_id(1) == 0 && (uint)(uint)sglid < 4 && token_pos_in_block < 4) {
                if (out_data_pitch == PAGED_ATTENTION_BLOCK_SIZE)
                    printf("  -- K token_num (%u) : sglid:(%u) in_off:(%u), out_off:(%u), out_pitch:(%u)\n", token_pos_in_block, sglid, in_data_offset, out_data_offset, out_data_pitch);
                else
                    printf("  -- V token_num (%u) : sglid:(%u) in_off:(%u), out_off:(%u), out_pitch:(%u)\n", token_pos_in_block, sglid, in_data_offset, out_data_offset, out_data_pitch);
            }
        }
        #endif  // ENABLE_DEBUG

        char2 res_vec = 0;
        uint packed_offset[((K_HEAD_SIZE+V_HEAD_SIZE) / SUBGROUP_SIZE) / PACK_SIZE] = {0, };
        unroll_for (uint i = 0; i < num_groups; i+=PACK_SIZE) {
            res_vec.s0 = convert_char_rte(input_data[i] * scale + zp);
            res_vec.s1 = (i+1 < num_groups) ? convert_char_rte(input_data[i+1] * scale + zp) : 0;

            packed_offset[i / PACK_SIZE] = out_data_offset + ((i / PACK_SIZE) * SUBGROUP_SIZE + sglid) * out_data_pitch;
            out_data[packed_offset[i / PACK_SIZE]] = cvt_int8x2_to_uint4x2(res_vec);

            #if ENABLE_DEBUG
            {
                OUTPUT_TYPE tmp = cvt_int8x2_to_uint4x2(res_vec);
                MAKE_VECTOR_TYPE(char, PACK_SIZE) buff = unpack_to_char(*(int4x2_t *)&tmp);
                result[i] = buff.s0;
                if (i+1 < num_groups) {
                    result[i+1] = buff.s1;
                }
            }
            #endif
        }

        #if ENABLE_DEBUG
        {
            uint orig_offset[K_HEAD_SIZE / SUBGROUP_SIZE] = {0, };
            for (uint i = 0; i < num_groups; i++) {
                OUTPUT_TYPE res = convert_char_rte(input_data[i] * scale + zp);

                orig_offset[i] = out_data_offset + (i * SUBGROUP_SIZE + sglid) * out_data_pitch;
                // [TEST]
                // out_data[orig_offset[i]] = res;
                orig_result[i] = res;
            }
        }
        #endif
    #else  // IS_INT4_COMPRESSED
        unroll_for (uint i = 0; i < num_groups; i++) {
            OUTPUT_TYPE res = convert_char_rte(input_data[i] * scale + zp);

            uint offset = out_data_offset + (i * SUBGROUP_SIZE + sglid) * out_data_pitch;
            out_data[offset] = res;
        }
    #endif  // IS_INT4_COMPRESSED

    INPUT0_TYPE* comp_ptr = out_data + comp_offset;

    if (sglid == 0) {
        comp_ptr[token_pos_in_block] = 1.0 / scale;
        comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + token_pos_in_block] = zp;
    }
}

#ifdef IS_KEY_BY_CHANNEL
#define COMP_K_OFFSET PAGED_ATTENTION_BLOCK_SIZE
#define NUM_HEAD_SIZE_GROUPS K_HEAD_SIZE / SUBGROUP_SIZE
inline void FUNC(quantize_and_save_by_channel_block_with_requantize)(__global const INPUT0_TYPE* in_data,
                                    const uint in_data_offset,
                                    const uint in_data_pitch,
                                    __global OUTPUT_TYPE* out_data,
                                    const uint out_data_offset,
                                    const uint out_data_pitch,
                                    const uint token_pos_in_block,
                                    const uint new_tokens_num,
                                    const uint sglid,
                                    const uint is_prefill_stage) {
    int head_size_offset = SUBGROUP_SIZE * get_group_id(2);
    int num_head_size_groups = is_prefill_stage ? NUM_HEAD_SIZE_GROUPS : NUM_HEAD_SIZE_GROUPS / NUM_K_HEAD_SIZE_PARTITIONS;
    for (int h_sub = 0; h_sub < num_head_size_groups; h_sub++) {
        const int hidden_idx = head_size_offset + h_sub * SUBGROUP_SIZE + sglid;
        const uint out_offset_per_wi = out_data_offset + hidden_idx * out_data_pitch;
        // Read original scale and zp
        INPUT0_TYPE* comp_ptr = (INPUT0_TYPE*) (&out_data[out_offset_per_wi + COMP_K_OFFSET]);
        const INPUT0_TYPE orig_scale = comp_ptr[0];
        const INPUT0_TYPE orig_zp = comp_ptr[1];
        INPUT0_TYPE max_value = INPUT0_VAL_MIN;
        INPUT0_TYPE min_value = INPUT0_VAL_MAX;
        // Read new input
        #define READ_SIZE 16
        #define OUT_DATA_VEC MAKE_VECTOR_TYPE(OUTPUT_TYPE, READ_SIZE)
        #define IN_DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_SIZE)
        #define VLOAD CAT(vload, READ_SIZE)
        IN_DATA_VEC cache_data_vec_decompressed;
        for (int j = 0; j < new_tokens_num; ++j) {
            INPUT0_TYPE new_token = BLOCK_READN(INPUT0_TYPE, 1, in_data, in_data_offset + j * K_HEAD_SIZE * KV_HEADS_NUM + hidden_idx);
            cache_data_vec_decompressed[token_pos_in_block + j] = new_token;
            max_value = fmax(max_value, new_token);
            min_value = fmin(min_value, new_token);
        }
        // Read a hidden dim of the previously quantized cache => decompress
        // TODO : current block size is 16 (same as PA block size),
        //        but when the block size becomes different, this part should be updated as well
        OUT_DATA_VEC prev_cache_data_vec = VLOAD(0, out_data + out_offset_per_wi);
        #undef READ_SIZE
        #undef VLOAD
        #undef DATA_VEC
        for (int j = 0; j < token_pos_in_block + new_tokens_num; ++j) {
            if (j < token_pos_in_block) {
                INPUT0_TYPE decompressed_cache_val = ((INPUT0_TYPE)prev_cache_data_vec[j] - orig_zp) * orig_scale;
                cache_data_vec_decompressed[j] = decompressed_cache_val;
            }
            max_value = fmax(max_value, cache_data_vec_decompressed[j]);
            min_value = fmin(min_value, cache_data_vec_decompressed[j]);
        }
        // requantize and store
        {
            #define ACCUMULATOR_TYPE float
            ACCUMULATOR_TYPE range = max_value - min_value;
            const ACCUMULATOR_TYPE min_range = fabs(max_value * 0.1f);
            if (range <= min_range) {
                // When the range is very small, expand the range to avoid zp overflow
                range += fmax(1.0f, min_range);
            }
            ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / range);
            ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
            INPUT0_TYPE scale = (INPUT1_TYPE)(scale_tmp);
            INPUT0_TYPE zp = (INPUT1_TYPE)(zp_tmp);
            #undef ACCUMULATOR_TYPE

            for (uint token = 0; token < token_pos_in_block + new_tokens_num; ++token) {
                OUTPUT_TYPE quantized = convert_char_rte(cache_data_vec_decompressed[token] * scale + zp);
                out_data[out_offset_per_wi + token] = quantized;
            }
            comp_ptr[0] = 1.0/scale;
            comp_ptr[1] = zp;
        }
    }
}


inline void FUNC(quantize_and_save_by_channel_block_with_requantize_int4)(__global const INPUT0_TYPE* in_data,
                                    const uint in_data_offset,
                                    const uint in_data_pitch,
                                    __global OUTPUT_TYPE* out_data,
                                    const uint out_data_offset,
                                    const uint out_data_pitch,
                                    const uint token_pos_in_block,
                                    const uint new_tokens_num,
                                    const uint sglid,
                                    const uint is_prefill_stage) {
    int head_size_offset = SUBGROUP_SIZE * get_group_id(2) * PACK_SIZE;
    int num_head_size_groups = is_prefill_stage ? NUM_HEAD_SIZE_GROUPS : NUM_HEAD_SIZE_GROUPS / (NUM_K_HEAD_SIZE_PARTITIONS / PACK_SIZE);

    int packed_head_size_offset = SUBGROUP_SIZE * (get_group_id(2));

    #if ENABLE_DEBUG
        OUTPUT_TYPE result_cache_int4[NUM_HEAD_SIZE_GROUPS][ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE] = {0, };
        uint index_cache_int4[NUM_HEAD_SIZE_GROUPS][ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE] = {0, };
        OUTPUT_TYPE result_comp_int4[NUM_HEAD_SIZE_GROUPS] = {0, };
        uint index_comp_int4[NUM_HEAD_SIZE_GROUPS] = {0, };
        OUTPUT_TYPE result_cache_int8[NUM_HEAD_SIZE_GROUPS][ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE] = {0, };
        uint index_cache_int8[NUM_HEAD_SIZE_GROUPS][ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE] = {0, };
        OUTPUT_TYPE result_comp_int8[NUM_HEAD_SIZE_GROUPS] = {0, };
        uint index_comp_int8[NUM_HEAD_SIZE_GROUPS] = {0, };
    #endif

    INPUT0_TYPE orig_scale[PACK_SIZE] = {0, };
    INPUT0_TYPE orig_zp[PACK_SIZE] = {0, };
    OUTPUT_TYPE orig_cache[ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE] = {0, };

    INPUT0_TYPE max_value[PACK_SIZE];
    INPUT0_TYPE min_value[PACK_SIZE];
    INPUT0_TYPE scale[PACK_SIZE];
    INPUT0_TYPE zp[PACK_SIZE];
    OUTPUT_TYPE buffer[ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE] = {0, };
    OUTPUT_TYPE current[ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE] = {0, };

    MAKE_VECTOR_TYPE(INPUT0_TYPE, 16) cache_data_vec_decompressed[NUM_HEAD_SIZE_GROUPS] = {0, };

    INPUT0_TYPE* prev_comp_ptr;
    for (int h_sub = 0; h_sub < num_head_size_groups; h_sub++) {
        const int hidden_idx = head_size_offset + h_sub * SUBGROUP_SIZE + sglid;
        // const uint out_offset_per_wi = out_data_offset + hidden_idx * out_data_pitch;

        // cache
        const uint order_in_packed = (h_sub % PACK_SIZE);
        // const int packed_hidden_idx = head_size_offset + (h_sub/PACK_SIZE) * SUBGROUP_SIZE + sglid;
        const int packed_hidden_idx = packed_head_size_offset + (h_sub/PACK_SIZE) * SUBGROUP_SIZE + sglid;
        const uint packed_out_offset_per_wi = out_data_offset + packed_hidden_idx * out_data_pitch;

        // Read original scale and zp
        if (order_in_packed == 0) {
            INPUT0_TYPE* comp_ptr = (INPUT0_TYPE*) (&out_data[packed_out_offset_per_wi + COMP_K_OFFSET]);
            orig_scale[order_in_packed] = comp_ptr[0];
            orig_zp[order_in_packed] = comp_ptr[1];
            orig_scale[1] = comp_ptr[2];
            orig_zp[1] = comp_ptr[3];
        }
        max_value[order_in_packed] = INPUT0_VAL_MIN;
        min_value[order_in_packed] = INPUT0_VAL_MAX;

        // Read new input
        #define READ_SIZE 16
        #define OUT_DATA_VEC MAKE_VECTOR_TYPE(OUTPUT_TYPE, READ_SIZE)
        #define IN_DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_SIZE)
        #define VLOAD CAT(vload, READ_SIZE)
        // IN_DATA_VEC cache_data_vec_decompressed[PACK_SIZE];
        for (int j = 0; j < new_tokens_num; ++j) {
            // Generate cache_data_vec_decompressed[token_pos_in_block ~ end] from Input (for order_in_packed)
            INPUT0_TYPE new_token = BLOCK_READN(INPUT0_TYPE, 1, in_data, in_data_offset + j * K_HEAD_SIZE * KV_HEADS_NUM + hidden_idx);

            // cache_data_vec_decompressed[token_pos_in_block + j] = new_token;
            cache_data_vec_decompressed[order_in_packed][token_pos_in_block + j] = new_token;

            max_value[order_in_packed] = fmax(max_value[order_in_packed], new_token);
            min_value[order_in_packed] = fmin(min_value[order_in_packed], new_token);
        }

        // Get values from cache
        {
            // Read a hidden dim of the previously quantized cache => decompress
            // TODO : current block size is 16 (same as PA block size),
            //        but when the block size becomes different, this part should be updated as well
            OUT_DATA_VEC prev_cache_data_vec;
            if (order_in_packed == 0)
                prev_cache_data_vec = VLOAD(0, out_data + packed_out_offset_per_wi);
            #undef READ_SIZE
            #undef VLOAD
            #undef DATA_VEC
            for (int j = 0; j < token_pos_in_block + new_tokens_num; ++j) {
                if (j < token_pos_in_block) {
                    if (order_in_packed == 0) {
                        // Generate cache_data_vec_decompressed[0~token_pos_in_block] from key-cache
                        char temp = prev_cache_data_vec[j];
                        MAKE_VECTOR_TYPE(char, PACK_SIZE) buff = unpack_to_char(*(int4x2_t *)&temp);

                        if (h_sub + 1 >= num_head_size_groups) {
                            cache_data_vec_decompressed[0][j] = ((INPUT0_TYPE)buff.s0 - orig_zp[0]) * orig_scale[0];
                            orig_cache[j] = buff.s1;
                        } else {
                            cache_data_vec_decompressed[0][j] = ((INPUT0_TYPE)buff.s0 - orig_zp[0]) * orig_scale[0];
                            cache_data_vec_decompressed[1][j] = ((INPUT0_TYPE)buff.s1 - orig_zp[1]) * orig_scale[1];
                        }
                    }
                }
                max_value[order_in_packed] = fmax(max_value[order_in_packed], cache_data_vec_decompressed[order_in_packed][j]);
                min_value[order_in_packed] = fmin(min_value[order_in_packed], cache_data_vec_decompressed[order_in_packed][j]);
            }
        }

        // requantize and store
        {
            #define ACCUMULATOR_TYPE float
            ACCUMULATOR_TYPE range = max_value[order_in_packed] - min_value[order_in_packed];
            range = (max_value[order_in_packed] == min_value[order_in_packed]) ? 0.001 : range;
            const ACCUMULATOR_TYPE min_range = fabs(max_value[order_in_packed] * 0.1f);
            if (range <= min_range) {
                // When the range is very small, expand the range to avoid zp overflow
                range += fmax(1.0f, min_range);
            }

            ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((INT4_RANGE) / range);
            ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value[order_in_packed] * scale_tmp) + INT4_MIN;
            scale[order_in_packed] = (INPUT1_TYPE)(scale_tmp);
            zp[order_in_packed] = (INPUT1_TYPE)(zp_tmp);
            #undef ACCUMULATOR_TYPE

            INPUT0_TYPE* comp_ptr = (INPUT0_TYPE*) (&out_data[packed_out_offset_per_wi + COMP_K_OFFSET]);
            // [TEST]
            // [0,1] 1st scale and zp, [2,3] 2nd scale and zp
            comp_ptr[2*order_in_packed] = 1.0/scale[order_in_packed];
            comp_ptr[2*order_in_packed+1] = zp[order_in_packed];
        }

        #if ENABLE_DEBUG
            // index_comp_int4[h_sub] = packed_out_offset_per_wi + COMP_K_OFFSET + order_in_packed * SCALE_ZP_SIZE_PER_TOKEN;
            index_comp_int4[h_sub] = packed_out_offset_per_wi + COMP_K_OFFSET;
            result_comp_int4[h_sub] = zp[order_in_packed];
        #endif

        for (uint token = 0; token < token_pos_in_block + new_tokens_num; ++token) {
            #if ENABLE_DEBUG
                OUTPUT_TYPE res = convert_char_rte(cache_data_vec_decompressed[1][token] * scale[1] + zp[1]);
            #endif

            if (order_in_packed == (PACK_SIZE - 1)) {
                char2 res_vec;
                res_vec.s0 = buffer[token];
                res_vec.s1 = convert_char_rte(cache_data_vec_decompressed[1][token] * scale[1] + zp[1]);

                out_data[packed_out_offset_per_wi + token] = cvt_int8x2_to_uint4x2(res_vec);

                #if ENABLE_DEBUG
                    OUTPUT_TYPE packed = cvt_int8x2_to_uint4x2(res_vec);
                    MAKE_VECTOR_TYPE(char, PACK_SIZE) tmp = unpack_to_char(*(int4x2_t *)&packed);

                    buffer[token] = res_vec.s0;
                    current[token] = tmp[1];

                    result_cache_int4[h_sub][token] = tmp[1];
                    index_cache_int4[h_sub][token] = packed_out_offset_per_wi + token;
                    index_cache_int4[h_sub-1][token] = packed_out_offset_per_wi + token;
                #endif
            } else {
                buffer[token] = convert_char_rte(cache_data_vec_decompressed[0][token] * scale[0] + zp[0]);

                #if ENABLE_DEBUG
                    result_cache_int4[h_sub][token] = buffer[token];
                #endif

                if (h_sub + 1 >= num_head_size_groups) {
                    char2 res_vec = 0;
                    res_vec.s0 = buffer[token];
                    res_vec.s1 = orig_cache[token];
                    // [TEST]
                    out_data[packed_out_offset_per_wi + token] = cvt_int8x2_to_uint4x2(res_vec);

                    #if ENABLE_DEBUG
                        current[token] = convert_char_rte(cache_data_vec_decompressed[1][token] * orig_scale[1] + orig_zp[1]);
                        OUTPUT_TYPE packed = cvt_int8x2_to_uint4x2(res_vec);
                        MAKE_VECTOR_TYPE(char, PACK_SIZE) tmp = unpack_to_char(*(int4x2_t *)&packed);

                        result_cache_int4[h_sub][token] = tmp[0];
                        index_cache_int4[h_sub][token] = packed_out_offset_per_wi + token;
                    #endif
                }
            }
        }

        #if ENABLE_DEBUG
        {
            if ((uint)get_global_id(0) == 0 && (uint)get_global_id(1) < 4 && h_sub < 6 && get_group_id(2) < 4) {
                    uint temp = order_in_packed + packed_head_idx;
                    printf("\t\t-- Requant: K & By-ch num_group(h_sub:%u / %u) sglid(%u) head_idx(%u) idx:block/head(%lu,%lu) : tokens_num(new_tk(%u)+tk_pos_blk(%u)) => input_off(%u), cache_offset_per_wi:(%u), out_offset_per_wi(%u) => value(%d:%d, %d:%d, %d:%d, %d:%d) scale(%.3f : %.3f) zp(%.2f: %.2f) orig_zp(%.2f: %.2f) => decomp(%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f)\n",
                            h_sub, num_head_size_groups, sglid, (uint)get_group_id(2), get_global_id(0), get_global_id(1),
                            new_tokens_num, token_pos_in_block, in_data_offset, packed_out_offset_per_wi, packed_out_offset_per_wi,
                            (int)buffer[0], (int)current[0], (int)buffer[1], (int)current[1], (int)buffer[2], (int)current[2], (int)buffer[3], (int)current[3],
                            // (int)buffer[4], (int)current[4], (int)buffer[5], (int)current[5], (int)buffer[6], (int)current[6], (int)buffer[7], (int)current[7],
                            (float)(1.0/scale[0]), (float)(1.0/scale[1]), (float)zp[0], (float)zp[1], (float)orig_zp[0], (float)orig_zp[1],
                            cache_data_vec_decompressed[temp][0], cache_data_vec_decompressed[temp][1], cache_data_vec_decompressed[temp][2], cache_data_vec_decompressed[temp][3],
                            cache_data_vec_decompressed[temp][4], cache_data_vec_decompressed[temp][5], cache_data_vec_decompressed[temp][6], cache_data_vec_decompressed[temp][7],
                            cache_data_vec_decompressed[temp][8], cache_data_vec_decompressed[temp][9], cache_data_vec_decompressed[temp][10], cache_data_vec_decompressed[temp][11]);
            }
        }

        if ((h_sub + packed_head_idx) % PACK_SIZE == (PACK_SIZE - 1)) {
            for (uint t = 0; t < token_pos_in_block + new_tokens_num; t++) {
                buffer[t] = 0;
                current[t] = 0;
                orig_cache[t] = 0;
            }
            zp[0] = zp[1] = 0;
            scale[0] = scale[1]= 0;
        }
        #endif
    }
}

inline void FUNC(quantize_and_save_by_channel_prefill)(__global const INPUT0_TYPE* in_data,
                                    const uint in_data_offset,
                                    const uint in_data_pitch,
                                    __global OUTPUT_TYPE* out_data,
                                    const uint out_data_offset,
                                    const uint tokens_num,
                                    //const uint token_start_pos_key,
                                    const uint sglid)  {
    uint out_offset = out_data_offset;
    #if IS_INT4_COMPRESSED
        // store the rest of the pair
        OUTPUT_TYPE buffer[PAGED_ATTENTION_BLOCK_SIZE] = {0, };
    #endif
    for (uint i = 0; i < NUM_HEAD_SIZE_GROUPS; i++) {
        uint key_in_offset_tmp = in_data_offset + i * SUBGROUP_SIZE;
        INPUT0_TYPE input_data[PAGED_ATTENTION_BLOCK_SIZE];
        INPUT0_TYPE max_value = INPUT0_VAL_MIN;
        INPUT0_TYPE min_value = INPUT0_VAL_MAX;
        // Read 16 tokens x 16 hidden
        unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
            input_data[token_num] = BLOCK_READN(INPUT0_TYPE, 1, in_data, key_in_offset_tmp + sglid);
            max_value = fmax(max_value, input_data[token_num]);
            min_value = fmin(min_value, input_data[token_num]);
            key_in_offset_tmp += in_data_pitch;
        }
        #define ACCUMULATOR_TYPE float
        ACCUMULATOR_TYPE range = max_value - min_value;
        const ACCUMULATOR_TYPE min_range = fabs(max_value * 0.1f);
        if (range <= min_range) {
            // When the range is very small, expand the range to avoid zp overflow
            range += fmax(1.0f, min_range);
        }
        ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((INT4_MAX - INT4_MIN) / range);
        ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + INT4_MIN;
        INPUT0_TYPE scale = (INPUT1_TYPE)(scale_tmp);
        INPUT0_TYPE zp = (INPUT1_TYPE)(zp_tmp);
        #undef ACCUMULATOR_TYPE

        // Quantize and save each hidden dim
        uint out_offset_per_wi = out_offset + sglid * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        // store comp_data
        INPUT0_TYPE* comp_ptr = (INPUT0_TYPE*) (&out_data[out_offset_per_wi + COMP_K_OFFSET]);
        // [TEST]
        #if IS_INT4_COMPRESSED
            const uint order_in_packed = i % PACK_SIZE;
            comp_ptr[2 * order_in_packed] = 1.0 / scale;
            comp_ptr[2 * order_in_packed + 1] = zp;
        #else
            comp_ptr[0] = 1.0 / scale;
            comp_ptr[1] = zp;
        #endif

        // Store quantized key
        #if IS_INT4_COMPRESSED
            for (uint token_num = 0; token_num < tokens_num; token_num++) {
                if (order_in_packed == (PACK_SIZE - 1)) {
                    char2 res_vec = 0;
                    res_vec.s0 = buffer[token_num];
                    res_vec.s1 = convert_char_rte(input_data[token_num] * scale + zp);
                    out_data[out_offset_per_wi] = cvt_int8x2_to_uint4x2(res_vec);

                    out_offset_per_wi++;
                } else {
                    buffer[token_num] = convert_char_rte(input_data[token_num] * scale + zp);

                    if (i + 1 >= NUM_HEAD_SIZE_GROUPS) {
                        char2 res_vec = 0;
                        res_vec.s0 = buffer[token_num];
                        res_vec.s1 = 0;
                        // [TEST]
                        out_data[out_offset_per_wi] = cvt_int8x2_to_uint4x2(res_vec);
                        out_offset_per_wi++;
                    }
                }
            }

            if (order_in_packed == (PACK_SIZE - 1)) {
                out_offset += (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE * SUBGROUP_SIZE);
            }
        #else
            for (uint token_num = 0; token_num < tokens_num; token_num++) {
                OUTPUT_TYPE res = convert_char_rte(input_data[token_num] * scale + zp);
                out_data[out_offset_per_wi] = res;

                out_offset_per_wi++;
            }
            out_offset += ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE * SUBGROUP_SIZE;
        #endif
    }
}
#endif  // IS_KEY_BY_CHANNEL

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUBGROUP_SIZE)))
KERNEL(pa_kv_cache_update)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* key_data,
    __global const INPUT1_TYPE* value_data,
    __global const INPUT2_TYPE* past_lens,
    __global const INPUT3_TYPE* block_indices,
    __global const INPUT4_TYPE* block_indices_begins,
    __global const INPUT5_TYPE* subsequence_begins,
    __global OUTPUT_TYPE* key_cache_data,
    __global OUTPUT1_TYPE* value_cache_data,
    const __global int* blocked_indexes_start,
    const __global int* blocked_indexes_end,
    const __global int* gws_seq_indexes_correspondence,
    const int is_prefill_stage
) {
    // If the the number of new tokens equals to the number of past_lens elements,
    // then it's the 2nd+ iteration
    const uint KEY_IN_STRIDE = KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM;
    const uint VAL_IN_STRIDE = KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM;
    #ifdef IS_KEY_BY_CHANNEL
    const int k_hidden_stride = ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
    #endif

    #if IS_INT4_COMPRESSED
        const uint adjusted_k_head_size = PACKED_ADJUSTED_K_HEAD_SIZE;
        const uint adjusted_v_head_size = PACKED_ADJUSTED_V_HEAD_SIZE;
        const uint k_head_size = PACKED_K_HEAD_SIZE;
        const uint v_head_size = PACKED_V_HEAD_SIZE;
    #else
        const uint adjusted_k_head_size = ADJUSTED_K_HEAD_SIZE;
        const uint adjusted_v_head_size = ADJUSTED_V_HEAD_SIZE;
        const uint k_head_size = K_HEAD_SIZE;
        const uint v_head_size = V_HEAD_SIZE;
    #endif

    // [DEBUG]
    #if ENABLE_DEBUG
    {
        bool is_first = false;
        bool is_key_by_channel = false;
        if ((uint)get_global_id(0) == 0 && (uint)get_global_id(1) == 0 && (uint)get_global_id(2) == 0) {
            is_first = true;
            #if IS_KEY_BY_CHANNEL
                is_key_by_channel = true;
            #endif
            printf(">>>>>>>>>>>>>>>>>>>>>> (is_prefill:%d) is_key_by_channel(%d)\n", is_prefill_stage, is_key_by_channel);
        }
    }
    #endif
    if (!is_prefill_stage) {
        // 2nd+ token
        const uint seq_idx = (uint)get_global_id(0);
        const uint head_idx = (uint)get_global_id(1);
        const uint sglid = (uint)get_local_id(2);

        const uint past_seq_len = past_lens[seq_idx];
        const uint current_token_pos_in_block = past_seq_len % PAGED_ATTENTION_BLOCK_SIZE;
        const uint seq_block_idx = block_indices_begins[seq_idx] + past_seq_len / PAGED_ATTENTION_BLOCK_SIZE;
        const uint block_idx = block_indices[seq_block_idx];

        uint key_in_offset = INPUT0_OFFSET + seq_idx * KEY_IN_STRIDE + head_idx * K_HEAD_SIZE;
        uint value_in_offset = INPUT1_OFFSET + seq_idx * VAL_IN_STRIDE + head_idx * V_HEAD_SIZE;

        #ifdef IS_KEY_BY_CHANNEL
        uint block_k_base_offset = block_idx * KV_HEADS_NUM * adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + head_idx * adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        #else // can it be shared?
        uint block_k_base_offset = block_idx * KV_HEADS_NUM * adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE + head_idx * adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
        #endif
        uint block_v_base_offset = block_idx * KV_HEADS_NUM * adjusted_v_head_size * PAGED_ATTENTION_BLOCK_SIZE + head_idx * adjusted_v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
        uint key_out_offset = block_k_base_offset + current_token_pos_in_block;
        uint value_out_offset = block_v_base_offset + current_token_pos_in_block * v_head_size;
#if !IS_KV_COMPRESSED
        #define READ_K_BLOCK_SIZE GENERATE_STAGE_K_BLOCK_SIZE
        for (uint head_idx_index = 0; head_idx_index < K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_K_BLOCK_SIZE) {
            #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_K_BLOCK_SIZE, ptr, offset);
            #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_K_BLOCK_SIZE)

            DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

            unroll_for (uint i = 0; i < READ_K_BLOCK_SIZE; i++) {
                uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                #if READ_K_BLOCK_SIZE == 1
                    key_cache_data[key_offset] = input_data;
                #else
                    key_cache_data[key_offset] = input_data[i];
                #endif
            }
        }

        #define READ_V_BLOCK_SIZE GENERATE_STAGE_V_BLOCK_SIZE
        for (uint head_idx_index = 0; head_idx_index < V_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_V_BLOCK_SIZE) {
            #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_V_BLOCK_SIZE, ptr, offset);
            #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_V_BLOCK_SIZE)

            DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + head_idx_index);

            unroll_for (uint i = 0; i < READ_V_BLOCK_SIZE; i++) {
                uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                #if READ_V_BLOCK_SIZE == 1
                    value_cache_data[value_offset] = input_data;
                #else
                    value_cache_data[value_offset] = input_data[i];
                #endif
            }
        }
#else // IS_KV_COMPRESSED
        #ifdef IS_KEY_BY_CHANNEL
            #if ENABLE_DEBUG && 0
            if ((sglid == 0 || sglid == 15) && block_idx < 4) {
                printf("\t>> is_prefill(false) block_idx(%u) head_idx(%u) sglid(%u) => current_token_pos_in_block(%u) tokens_num(%u) key_out_offset(%u) block_k_base_offset(%u)\n",
                        block_idx, head_idx, sglid, current_token_pos_in_block, 1, key_out_offset, block_k_base_offset);
            }
            #endif
        // key by channel
        {
            #if IS_INT4_COMPRESSED
                FUNC_CALL(quantize_and_save_by_channel_block_with_requantize_int4)(key_data,
                                                                            key_in_offset,
                                                                            KEY_IN_STRIDE,
                                                                            key_cache_data,
                                                                            block_k_base_offset,
                                                                            k_hidden_stride,
                                                                            current_token_pos_in_block,
                                                                            1,
                                                                            sglid,
                                                                            0);
            #else
                FUNC_CALL(quantize_and_save_by_channel_block_with_requantize)(key_data,
                                                                            key_in_offset,
                                                                            KEY_IN_STRIDE,
                                                                            key_cache_data,
                                                                            block_k_base_offset,
                                                                            k_hidden_stride,
                                                                            current_token_pos_in_block,
                                                                            1,
                                                                            sglid,
                                                                            0);
            #endif
        }
        // value per token
        if (get_group_id(2) == 0) {
            const uint comp_v_offset = block_v_base_offset + v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
            INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
            FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1, comp_v_offset,
                current_token_pos_in_block, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
        }
        #else
        // IS_KEY_BY_CHANNEL false
        {
            const uint comp_k_offset = block_k_base_offset + k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
            // key processing
            INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
            FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE, comp_k_offset,
                current_token_pos_in_block, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
        }
        {
            const uint comp_v_offset = block_v_base_offset + v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
            INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
            FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1, comp_v_offset,
                current_token_pos_in_block, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
        }
        #endif
#endif // IS_KV_COMPRESSED
    } else {
        // 1st token
        const uint block_idx = get_global_id(0);
        const uint head_idx = get_global_id(1);
        const uint sglid = get_global_id(2);

        const uint subsequence_idx = gws_seq_indexes_correspondence[block_idx];
        const uint subsequence_begin_idx = subsequence_begins[subsequence_idx];

        const uint block_start_pos = blocked_indexes_start[block_idx];
        const uint block_end_pos = blocked_indexes_end[block_idx];
        const uint tokens_num = block_end_pos - block_start_pos;
        const uint past_len = past_lens[subsequence_idx];
        const uint token_start_pos_key = (past_len + block_start_pos - subsequence_begin_idx) % PAGED_ATTENTION_BLOCK_SIZE;
        const uint token_start_pos_val = (past_len + block_start_pos - subsequence_begin_idx) % PAGED_ATTENTION_BLOCK_SIZE;

        uint key_in_offset = INPUT0_OFFSET + block_start_pos * KEY_IN_STRIDE + head_idx * K_HEAD_SIZE;

        uint value_in_offset = INPUT1_OFFSET + block_start_pos * VAL_IN_STRIDE + head_idx * V_HEAD_SIZE;

        const uint current_block_idx = (past_len + block_start_pos - subsequence_begin_idx) / PAGED_ATTENTION_BLOCK_SIZE;

        const uint block_offset = block_indices_begins[subsequence_idx] + current_block_idx;

        #if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            uint block_k_base_offset = block_indices[block_offset] * KV_HEADS_NUM * adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE +
                                    head_idx * adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
            uint key_out_offset = block_k_base_offset;
        #else
            uint block_k_base_offset = block_indices[block_offset] * KV_HEADS_NUM * adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE +
                                    head_idx * adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
            uint key_out_offset = block_k_base_offset;
            const uint comp_k_offset = block_k_base_offset + k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
            key_out_offset += token_start_pos_key;
        #endif

        #if ENABLE_DEBUG && 0
        if ((sglid == 0 || sglid == 15) && block_idx < 4) {
            printf(">>>>>>>>>>>> block_idx(%u) head_idx(%u) sglid(%u) => token_start_pos_key(%u) tokens_num(%u) key_out_offset(%u) => KV_HEADS_NUM(%u) ADJUSTED_K_HEAD_SIZE(%u(%u)) ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE(%u(%u)) block_xxx_pos(end:%u,start:%u)\n",
                    block_idx, head_idx, sglid, token_start_pos_key, tokens_num, key_out_offset,
                    (uint)KV_HEADS_NUM, (uint)ADJUSTED_K_HEAD_SIZE, (uint)K_HEAD_SIZE, (uint)ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE, (uint)PAGED_ATTENTION_BLOCK_SIZE, block_end_pos, block_start_pos);
        }
        #endif

        uint block_v_base_offset = block_indices[block_offset] * KV_HEADS_NUM * adjusted_v_head_size * PAGED_ATTENTION_BLOCK_SIZE +
                                 head_idx * adjusted_v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
        const uint comp_v_offset = block_v_base_offset + v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
        uint value_out_offset = block_v_base_offset;
        value_out_offset += token_start_pos_val * v_head_size;

        if (tokens_num == PAGED_ATTENTION_BLOCK_SIZE) {
        // block is full
        #if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            // Key by channel
            if (token_start_pos_key != 0) {
                #if IS_INT4_COMPRESSED
                FUNC_CALL(quantize_and_save_by_channel_block_with_requantize_int4)(key_data,
                                                                            key_in_offset,
                                                                            KEY_IN_STRIDE,
                                                                            key_cache_data,
                                                                            block_k_base_offset,
                                                                            k_hidden_stride,
                                                                            token_start_pos_key,
                                                                            tokens_num,
                                                                            sglid,
                                                                            1);
                #else
                // mixed mode => need requantize with prev tokens
                FUNC_CALL(quantize_and_save_by_channel_block_with_requantize)(key_data,
                                                                            key_in_offset,
                                                                            KEY_IN_STRIDE,
                                                                            key_cache_data,
                                                                            block_k_base_offset,
                                                                            k_hidden_stride,
                                                                            token_start_pos_key,
                                                                            tokens_num,
                                                                            sglid,
                                                                            1);
                #endif
            } else {
                FUNC_CALL(quantize_and_save_by_channel_prefill)(key_data,
                                                                key_in_offset,
                                                                KEY_IN_STRIDE,
                                                                key_cache_data,
                                                                key_out_offset,
                                                                tokens_num,
                                                                sglid);
            }
            // Value per token
            unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
                INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                    comp_v_offset, token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += v_head_size;
            }
        #else // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            unroll_for (uint token_num = 0; token_num < PAGED_ATTENTION_BLOCK_SIZE; token_num++) {
            #if !IS_KV_COMPRESSED
            {
                uint head_idx_index = 0;

                #define READ_BLOCK_SIZE 8
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 4
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 2
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 1
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data;
                    }
                }
            }
            {

                uint v_head_idx_index = 0;

                #define READ_BLOCK_SIZE 8
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 4
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 2
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }


                #define READ_BLOCK_SIZE 1
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data;
                    }
                }
            }
            #else // IS_KV_COMPRESSED
            #if ENABLE_DEBUG
            {
                if ((sglid == 0 || sglid == 15) &&
                    (uint)get_global_id(0) == 0 && (uint)get_global_id(1) == 0 && (uint)get_global_id(2) == 0) {
                    printf("\t>> comp_v_offset(%u) : block_v_base_offset(%u) v_head_size(%u) V_HEAD_SIZE(%u) PAGED_ATTENTION_BLOCK_SIZE(%u) \n",
                                comp_v_offset, block_v_base_offset, (uint)v_head_size, (uint)V_HEAD_SIZE, (uint)PAGED_ATTENTION_BLOCK_SIZE);
                }
            }
            #endif
            {
                // Key per token
                INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE,
                    comp_k_offset, token_num, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
            }
            {
                // Value per token
                INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                    comp_v_offset, token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
            }
            #endif // IS_KV_COMPRESSED
                key_in_offset += (KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM);
                key_out_offset += 1;
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += v_head_size;
            }
        #endif // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
        } else {
        // block with leftover tokens (tokens_num != PAGED_ATTENTION_BLOCK_SIZE)
        #if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            // key processing by channel
            if (token_start_pos_key != 0) {
                // mixed mode => need requantize with prev tokens
                #if IS_INT4_COMPRESSED
                FUNC_CALL(quantize_and_save_by_channel_block_with_requantize_int4)(key_data,
                #else
                FUNC_CALL(quantize_and_save_by_channel_block_with_requantize)(key_data,
                #endif
                                                                            key_in_offset,
                                                                            KEY_IN_STRIDE,
                                                                            key_cache_data,
                                                                            block_k_base_offset,
                                                                            k_hidden_stride,
                                                                            token_start_pos_key,
                                                                            tokens_num,
                                                                            sglid,
                                                                            1);
            } else {
                FUNC_CALL(quantize_and_save_by_channel_prefill)(key_data,
                                                                key_in_offset,
                                                                KEY_IN_STRIDE,
                                                                key_cache_data,
                                                                key_out_offset,
                                                                tokens_num,
                                                                sglid);
            }

            // value processing per token
            for (uint token_num = 0; token_num < tokens_num; token_num++) {
                INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                    comp_v_offset, token_start_pos_val + token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += v_head_size;
            }
        #else // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            #if ENABLE_DEBUG
                printf("\t>> comp_k_offset(%u) : block_k_base_offset(%u) K_HEAD_SIZE(%u) PAGED_ATTENTION_BLOCK_SIZE(%u) \n",
                            comp_k_offset, block_k_base_offset, (uint)K_HEAD_SIZE, (uint)PAGED_ATTENTION_BLOCK_SIZE);
            #endif
            for (uint token_num = 0; token_num < tokens_num; token_num++) {
                uint head_idx_index = 0;

            #if !IS_KV_COMPRESSED
                #define READ_BLOCK_SIZE 1
                #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)
                for (uint head_idx_index = 0; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data;
                    }
                }

                for (uint head_idx_index = 0; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data;
                    }
                }

            #else // IS_KV_COMPRESSED
                {
                    // key processing
                    INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
                    FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE,
                        comp_k_offset, token_start_pos_key + token_num, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                }
                {
                    // value processing
                    INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                    FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                        comp_v_offset, token_start_pos_val + token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                }
            #endif // IS_KV_COMPRESSED
                key_in_offset += (KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM);
                key_out_offset += 1;
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += v_head_size;
            }
        #endif // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
        }
    }
}
