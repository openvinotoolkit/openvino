// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/imad.cl"

// JIT Parameters:
// SIMD         - sub-group size/simd width, one of {8, 16};
// TILE_B       - number of batches processed by each work-item;
// TILE_OFM     - number of output features calculated by work-item, one of {1, 2, 4, 8};
// TILE_IFM     - number of input features loaded from input by work-item, one of {1, 2, 4, 8};
// TILE_K       - number of input features loaded from weights, one of {1, 2, 4, 8};
// TILE_K_OFM   - must be equal to TILE_OFM * TILE_K and less or equal to 8;
// DISPATCH_FSV - output coordinates for each sub-group are calculated from linearized coordinates
// DISPATCH_BSV   as if they laid in bs_fs_bsv_fsv format, these macros describe fsv and bsv factors;

// Verify JIT parameters.
#if SIMD != 8 && SIMD != 16
#   error "fully_connected_gpu_bf_tiled.cl - SIMD must be one of {8, 16}"
#endif

#if TILE_OFM != 1 && TILE_OFM != 2 && TILE_OFM != 4 && TILE_OFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_OFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_IFM != 1 && TILE_IFM != 2 && TILE_IFM != 4 && TILE_IFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_IFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_K != 1 && TILE_K != 2 && TILE_K != 4 && TILE_K != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4, 8}"
#endif

#if TILE_K_OFM != (TILE_K * TILE_OFM) || TILE_K_OFM > 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be equal to TILE_K * TILE_OFM and at most 8"
#endif

#if COMPRESSED_WEIGHTS_INT4
#   if TILE_K_OFM != TILE_K_OFM_PACKED * 2
#       error "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be divisible by 2 for 4-bit compressed case"
#   endif
#endif

// Macros for vectorized types.
#define INPUT_VEC_TYPE             MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_IFM)
#define ACCUMULATOR_VEC_TYPE       MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define FILTER_VEC_TYPE            MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_K_OFM)
#define FILTER_PACKED_VEC_TYPE     MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_K_OFM_PACKED)
#define BIAS_VEC_TYPE              MAKE_VECTOR_TYPE(BIAS_TYPE, TILE_OFM)
#define OUTPUT_VEC_TYPE            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_OFM)
#define ACTIVATION_VEC_TYPE        MAKE_VECTOR_TYPE(ACTIVATION_TYPE, TILE_OFM)
#define TO_OUTPUT_VEC_TYPE(x)      CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_ACTIVATION_VEC_TYPE(x)  CAT(convert_, ACTIVATION_VEC_TYPE)(x)
#define TO_FILTER_VEC_TYPE(x)      CAT(convert_, FILTER_VEC_TYPE)(x)
#define TO_ACCUMULATOR_VEC_TYPE(x) CAT(convert_, ACCUMULATOR_VEC_TYPE)(x)

#define INPUT_BLOCK_READ(ptr, offset)        BLOCK_READN(INPUT0_TYPE,  TILE_IFM, ptr, offset)
#define FILTER_BLOCK_READ(ptr, offset)       BLOCK_READN(FILTER_TYPE, TILE_K_OFM_PACKED, ptr, offset)
#define BIAS_BLOCK_READ(ptr, offset)         BLOCK_READN(BIAS_TYPE, TILE_OFM, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, TILE_OFM, ptr, offset, val)

#define SLM_FILTER_VEC          MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define SLM_FILTER_PACKED_VEC   MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE)
#define SLM_FILTER_UNPACKED_VEC MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, FILTER_ELEMENTS_PER_LOAD)

// Dyc Quantize
#define INPUT_LOAD_SIZE                     4
#define DQ_TYPE                             char
#define DQ_VEC_TYPE                         MAKE_VECTOR_TYPE(DQ_TYPE, TILE_IFM)
#define DQ_SLM_FILTER_VEC                   MAKE_VECTOR_TYPE(DQ_TYPE, TILE_OFM)
#define DQ_SLM_FILTER_PACKED_VEC            MAKE_VECTOR_TYPE(DQ_TYPE, FILTER_LOAD_BLOCK_SIZE)
#define DQ_SLM_FILTER_UNPACKED_VEC          MAKE_VECTOR_TYPE(DQ_TYPE, FILTER_ELEMENTS_PER_LOAD)
#define DQ_FILTER_VEC_TYPE                  MAKE_VECTOR_TYPE(DQ_TYPE, TILE_K_OFM)

#define TO_DQ_TYPE(x)                       CAT(CAT(convert_, DQ_TYPE),_sat)(x)
#define TO_DQ_VEC_TYPE(x)                   CAT(convert_, DQ_VEC_TYPE)(x)
#define TO_DQ_SLM_FILTER_UNPACKED_VEC(x)  CAT(convert_, DQ_SLM_FILTER_UNPACKED_VEC)(x)
#define TO_DQ_FILTER_VEC_TYPE(x)            CAT(convert_, DQ_FILTER_VEC_TYPE)(x)

#define UNPACK_MIXED_INT4x2(target_type, value) CAT(unpack_mixed_to_, target_type)(value)

#define AS_TYPE_N_(type, n, x)  as_##type##n(x)
#define AS_TYPE_N(type, n, x)   AS_TYPE_N_(type, n, x)
#define AS_DQ_TYPE_4(x)         AS_TYPE_N(DQ_TYPE, INPUT_LOAD_SIZE, x)

// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * TILE_OUT_B_PITCH) % 16 == 0 && (OUTPUT_TYPE_SIZE * OUTPUT_OFFSET) % 16 == 0)

#if !REALIGN_FP16_OFFSET
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_SIZE_Y
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_ELEMENTS_COUNT
#   endif
#else
// For REALIGN_FP16_OFFSET one feature is processed separately before entering main loop to correct alignment.
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_SIZE_Y - 1)
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_ELEMENTS_COUNT - 1)
#   endif
#endif

#if OUTPUT_3D
#   define INPUT_ELEMENTS_COUNT INPUT0_SIZE_Y
#else
#   define INPUT_ELEMENTS_COUNT INPUT0_ELEMENTS_COUNT
#endif

#if IS_DYNAMIC && COMPRESSED_WEIGHTS_INT4
#pragma disable_includes_optimization
#define FORCED_TILE_B 1
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 2
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 3
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 4
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 5
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 6
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 7
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B
#pragma enable_includes_optimization
#endif

inline void FUNC(fc_bf_tiled_kernel_default)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM  // DECOMPRESSION_SCALE_TERM == 1
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR  // DECOMPRESSION_ZP_SCALAR == 1
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if USE_SLM
    // [Packing weight]
    // , __local DQ_TYPE* dq_wei_local_mem
    , __local int* dq_wei_local_mem
#endif
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if USE_SLM
    uint gid = (uint)get_group_id(0);
    uint local_id = (uint)get_local_id(2);
#else
    uint gid = (uint)get_group_id(0);
#endif
    uint sglid = (uint)get_sub_group_local_id();

    // Dispatch as bs_fs_bsv_fsv, where bsv = DISPATCH_BSV and fsv = DISPATCH_FSV.
    // This allows more fine grained control over dispatch order than using work-groups and
    // avoids requirement of threads being available for whole work-group.
    // It could hovewer have some drawbacks like not providing physical locality or not using
    // full dispatch pipeline.
    uint feature_mini_block = gid % DISPATCH_FSV;
    uint batch_mini_block = gid / DISPATCH_FSV % DISPATCH_BSV;
    uint feature_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV) % (CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);
    uint batch_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV * CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);

#if USE_SLM
    uint out_f = gid * (TILE_OFM * SIMD);  // out_f = gid * (2 * 16)
    uint out_b = LWS_BATCHES * TILE_B * (uint)get_group_id(2) + local_id * TILE_B;
#else
    uint out_f = (feature_mega_block * DISPATCH_FSV + feature_mini_block) * (TILE_OFM * SIMD);
    uint out_b = ((batch_mega_block * DISPATCH_BSV + batch_mini_block) * TILE_B);
#endif

    ACCUMULATOR_VEC_TYPE    acc[TILE_B] = { };              // MAKE_VECTOR_TYPE(half, 2) acc[8] = { };
    INPUT_VEC_TYPE          in_0[TILE_B] = { };             // MAKE_VECTOR_TYPE(half, 2) in_0[8] = { };

    MAKE_VECTOR_TYPE(INPUT0_TYPE, INPUT_LOAD_SIZE)  tiled_input_0[TILE_B/2] = { };     // Packing 4 elements of char inputs to 1 integer
    // MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE)      dq_in_0_to_int[TILE_B/2] = { };  // Packing target : quantized input (4x4)
    INPUT0_TYPE                       max[TILE_B/2] = { };

    int packed_in_0[TILE_B/2] = { }; // Packing char4 to int

#if !USE_SLM
    #if DECOMPRESSION_SCALE_POST_OP
        DQ_FILTER_VEC_TYPE wei = 0;
    #else
        FILTER_VEC_TYPE wei = 0;
    #endif
#endif

    uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
#if COMPRESSED_WEIGHTS_INT4  // COMPRESSED_WEIGHTS_INT4 == 1
    uint weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2);
#else
    uint weights_offset = out_f * INPUT_ELEMENTS_COUNT;
#endif

// Temporal comment out
#if 0
#if COMPRESSED_WEIGHTS && DECOMPRESSION_SCALE_GROUPS_NUM == 1
    #if DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_scale = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_SCALE_TYPE, TILE_OFM, decompression_scale, out_f));
    #elif DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_scale = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_SCALE_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_scale))[of] = decompression_scale[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_scale = decompression_scale[0];
    #endif

    ACCUMULATOR_TYPE* d_scales = (ACCUMULATOR_TYPE*)(&d_scale);
#endif
#endif

#if COMPRESSED_WEIGHTS && DECOMPRESSION_ZP_TERM && DECOMPRESSION_ZP_GROUPS_NUM == 1 && !DECOMPRESSION_ZP_SCALAR
    #if DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_zp = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_ZP_TYPE, TILE_OFM, decompression_zp, out_f));
    #elif DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_zp = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_ZP_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_zp))[of] = decompression_zp[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_zp = decompression_zp[0];
    #endif
    ACCUMULATOR_TYPE* d_zps = (ACCUMULATOR_TYPE*)(&d_zp);
#endif

#if 0
#if REALIGN_FP16_OFFSET
    // For fp16 we need to ensure that all block reads are aligned to 4 byte (2 words) boundary.
    // To do this solve first input feature separately.
    {
        INPUT0_TYPE tmp_input = input[input_offset + get_sub_group_local_id() % TILE_B * TILE_IN_B_PITCH];
        ACCUMULATOR_VEC_TYPE tmp_wei = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(FILTER_TYPE, TILE_OFM, weights, weights_offset));
        #if COMPRESSED_WEIGHTS
            tmp_wei = (tmp_wei - d_zp) * d_scale;
        #endif
        unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
            acc[bi] = _sub_group_shuffle(tmp_input, bi) * tmp_wei;
        }

        weights_offset += TILE_OFM * SIMD;
        input_offset += 1;
    }
#endif
#endif

    // =====================================================================================================================================
    // Main computation loop
    uint iterations = MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD);  // desc : iterations = 4096 / (2 * 16);
    // Packing index
    uint idx_sglid = (sglid * TILE_K) % 32;       // same index for sglid 0~7 : to tile_k direction
    uint batch_sglid = (sglid * TILE_K) / 32;     // 0 to 1 : to batch direction

    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Packing : Get 4x4 integer vector (to be 4(b)x4(k) char vector -> 4x1 vector)
        uint input_offset_tmp = input_offset + (idx_sglid + batch_sglid * TILE_IN_B_PITCH);
        for (uint bi = 0; bi < TILE_B/2; ++bi) {
            tiled_input_0[bi] = vload4(0, &input[input_offset_tmp]);

            // [TEMP]
            // input_offset_tmp += TILE_IN_B_PITCH;
            input_offset_tmp += (TILE_IN_B_PITCH * 2);
        }

        // Load input.
#if 0
        #define LOAD_IN_0(bi) do {                                    \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);     \
                input_offset += TILE_IN_B_PITCH;                      \
            } while (false)

        CONST_LOOP(TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
#else
        input_offset += TILE_IN_B_PITCH * 8;
#endif

        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * TILE_B;
        // NOTE: Manually unrolling multiplication loop leads to lower register pressure and allows for bigger block sizes,
        //       but significantly degrades readability and generality of code.
        //       It doesn't also show noticable performance improvement on tested configurations.
        #if DECOMPRESSION_SCALE_POST_OP
            // MAKE_VECTOR_TYPE(int, TILE_OFM) acc_tmp[TILE_B] = { };
            MAKE_VECTOR_TYPE(int, TILE_B) acc_tmp[TILE_OFM] = { };

        #endif

        #if 1
            barrier(CLK_LOCAL_MEM_FENCE);
            // Quantizing for loaded input using max value
            MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) de_quantize_scale = 1;
            MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) quantize_scale = 1;
            MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) dq_max_input;
            unroll_for (uint bi = 0; bi < TILE_B/2; ++bi) {
                max[bi] = fmax(fmax(fabs(tiled_input_0[bi][0]), fabs(tiled_input_0[bi][1])), fmax(fabs(tiled_input_0[bi][2]), fabs(tiled_input_0[bi][3])));
                dq_max_input[bi] = sub_group_reduce_max(max[bi]);
            }
            quantize_scale = 128 / dq_max_input;
            de_quantize_scale = dq_max_input / 128;
            // Packing 4 of converted inputs to integer type
            unroll_for (uint bi = 0; bi < TILE_B/2; ++bi) {
                packed_in_0[bi] = as_int(CAT(convert_, MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE))(tiled_input_0[bi] * quantize_scale[bi]));
            }
        #endif

        // Handle compressed weight
        // ACCUMULATOR_TYPE max_weight = 0;
        // USE_SLM == 1 && COMPRESSED_WEIGHTS_INT4 == 1
        #if USE_SLM && COMPRESSED_WEIGHTS_INT4
            #if TILE_OFM != 2
            #error "FC bf_tiled kernel: can't use SLM optimization with TILE_OFM != 2"
            #endif

            // Skip first barrier synchronization if there is only single outer loop iteration.
            #if MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD) > 1
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif

            // // Dump
            // {
            //     if (get_group_id(0) == 0 && get_group_id(2) == 0 && ni == 0 &&
            //         /*get_local_id(0) == 0 && */get_local_id(2) == 0) {
            //         for (uint bi = 0; bi < TILE_B/2; ++bi) {
            //             MAKE_VECTOR_TYPE(DQ_TYPE, 4) temp = AS_DQ_TYPE_4(packed_in_0[bi]);            //             if (get_sub_group_local_id() < 8) {
            //                 printf(" -- DQ : sub_grp_id (%d) K_idx(%d) Batch(%d) TILE_K direction: dq_in_0_to_int[0](%d),  dq_in_0_to_int[1](%d),  dq_in_0_to_int[3](%d),  dq_in_0_to_int[4](%d) \n",
            //                     get_sub_group_local_id(), (int)idx_sglid, (int)bi+batch_sglid, dq_in_0_to_int[bi][0], dq_in_0_to_int[bi][1], dq_in_0_to_int[bi][2], dq_in_0_to_int[bi][3]);
            //             } else {
            //                 printf(" == DQ : sub_grp_id (%d) K_idx(%d) Batch(%d) TILE_K direction: dq_in_0_to_int[0](%d),  dq_in_0_to_int[1](%d),  dq_in_0_to_int[3](%d),  dq_in_0_to_int[4](%d) \n",
            //                     get_sub_group_local_id(), (int)idx_sglid, (int)bi+batch_sglid, dq_in_0_to_int[bi][0], dq_in_0_to_int[bi][1], dq_in_0_to_int[bi][2], dq_in_0_to_int[bi][3]);
            //             }
            //         }
            //     }
            // }

            // [Packing weight]
            // __local DQ_SLM_FILTER_VEC* char_slm_weight = (__local DQ_SLM_FILTER_VEC*)dq_wei_local_mem;
            __local int* char_slm_weight = (__local int*)dq_wei_local_mem;

            uint weights_idx = weights_offset + local_id * SIMD * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE;
            // uint wei_local_idx = local_id * SIMD * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE + sglid * 4;
            // [Packing weight]
            uint wei_local_idx = local_id * SIMD * FILTER_LOAD_ITERS * (FILTER_LOAD_BLOCK_SIZE/2) + sglid * 2;
            // [Weight along local_id]
            // uint wei_local_idx = sglid * (FILTER_LOAD_BLOCK_SIZE/2) * FILTER_LOAD_ITERS * 8 + local_id * 2;

            // Temporally block : use cpp code
            #if 0
            unroll_for(uint load_iter = 0; load_iter < FILTER_LOAD_ITERS; ++load_iter) {
                SLM_FILTER_PACKED_VEC wei_packed = BLOCK_READN(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE, weights, weights_idx);
                #if DECOMPRESSION_SCALE_POST_OP
                    DQ_SLM_FILTER_UNPACKED_VEC dq_wei_unpacked = UNPACK_MIXED_INT4x2(DQ_TYPE, *((INT4_PACKED_TYPE_PRELOAD*)&wei_packed));
                    DQ_TYPE* dq_w = (DQ_TYPE*)(&dq_wei_unpacked);
                #else
                    SLM_FILTER_UNPACKED_VEC wei_unpacked = UNPACK_INT4x2(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE_PRELOAD*)&wei_packed));
                    ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei_unpacked);
                #endif
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    unroll_for(uint kii = 0; kii < FILTER_LOAD_BLOCK_SIZE; ++kii) {
                        // const uint w_idx = kii * TILE_OFM + fi;
                        const uint w_idx = fi * FILTER_LOAD_BLOCK_SIZE + kii;
                        const uint offset_ofm = out_f + fi*SIMD + sglid;
                        const uint offset_ifm = ni * TILE_IFM * SIMD + local_id * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE + load_iter * FILTER_LOAD_BLOCK_SIZE + kii;
                        // Get ds : half ds = decompression_scale[scale_offset];
                        #if !DECOMPRESSION_SCALE_POST_OP
                            #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                                const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH  +
                                                          (offset_ifm / DECOMPRESSION_SCALE_GROUP_SIZE) * DECOMPRESSION_SCALE_FEATURE_PITCH;
                                ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                            #else
                                ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE ds = ACCUMULATOR_VAL_ONE;
                        #endif

                        // Get dzp : DECOMPRESSION_ZP_VALUE == as_float(0x41000000);
                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                       (offset_ifm / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif

                        #if DECOMPRESSION_SCALE_POST_OP
                            dq_w[w_idx] = (dq_w[w_idx]- TO_DQ_TYPE(dzp));
                        #else
                            w[w_idx] = (w[w_idx] - dzp) * ds;
                        #endif
                    }
                }

                #define STORE_TO_SLM(vec2) char_slm_weight[wei_local_idx] = vec2; wei_local_idx += SIMD;

                #if FILTER_LOAD_BLOCK_SIZE == 2
                    STORE_TO_SLM(dq_wei_unpacked.s01);
                    STORE_TO_SLM(dq_wei_unpacked.s23);
                #elif FILTER_LOAD_BLOCK_SIZE == 4
                    STORE_TO_SLM(dq_wei_unpacked.s01);
                    STORE_TO_SLM(dq_wei_unpacked.s23);
                    STORE_TO_SLM(dq_wei_unpacked.s45);
                    STORE_TO_SLM(dq_wei_unpacked.s67);
                #elif FILTER_LOAD_BLOCK_SIZE == 8
                    STORE_TO_SLM(dq_wei_unpacked.s01);
                    STORE_TO_SLM(dq_wei_unpacked.s23);
                    STORE_TO_SLM(dq_wei_unpacked.s45);
                    STORE_TO_SLM(dq_wei_unpacked.s67);
                    STORE_TO_SLM(dq_wei_unpacked.s89);
                    STORE_TO_SLM(dq_wei_unpacked.sab);
                    STORE_TO_SLM(dq_wei_unpacked.scd);
                    STORE_TO_SLM(dq_wei_unpacked.sef);
                #else
                    #error "FC bf_tiled kernel: unsupported FILTER_LOAD_BLOCK_SIZE for SLM kernel"
                #endif

                #undef STORE_TO_SLM

                weights_idx += SIMD * FILTER_LOAD_BLOCK_SIZE;
            }
            #endif  // BLOCKED

            #if 1
            __attribute__((opencl_unroll_hint)) for (uint load_iter = 0; load_iter < 1; ++load_iter) {
                uchar4 wei_packed = as_uchar4(_sub_group_block_read_uc4((const __global uchar *)(weights) + (weights_idx)));

                char8 dq_wei_unpacked = unpack_mixed_to_char(*((uint4x8_t *)&wei_packed));
                char *dq_w = (char *)(&dq_wei_unpacked);

                // Dynamic Quantizing should use DECOMPRESSION_SCALE_POST_OP
                // __attribute__((opencl_unroll_hint)) for (uint fi = 0; fi < 2; ++fi) {
                //     __attribute__((opencl_unroll_hint)) for (uint kii = 0; kii < 4; ++kii) {
                //         const uint w_idx = fi * 4 + kii;
                //         // const uint offset_ofm = out_f + fi * 16 + sglid;
                //         // const uint offset_ifm = ni * 2 * 16 + local_id * 1 * 4 + load_iter * 4 + kii;
                //         half ds = 1.0h;
                //         half dzp = as_float(0x0);
                //         dq_w[w_idx] = (dq_w[w_idx] - convert_char_sat(dzp));
                //     }
                // }

                /*
                // char_slm_weight[wei_local_idx] = dq_wei_unpacked.s01;
                // wei_local_idx += 16;
                // char_slm_weight[wei_local_idx] = dq_wei_unpacked.s23;
                // wei_local_idx += 16;
                // char_slm_weight[wei_local_idx] = dq_wei_unpacked.s45;
                // wei_local_idx += 16;
                // char_slm_weight[wei_local_idx] = dq_wei_unpacked.s67;
                // wei_local_idx += 16;
                // weights_idx += 16 * 4;
                char_slm_weight[wei_local_idx] = dq_wei_unpacked.s01;
                char_slm_weight[wei_local_idx+1] = dq_wei_unpacked.s23;
                char_slm_weight[wei_local_idx+2] = dq_wei_unpacked.s45;
                char_slm_weight[wei_local_idx+3] = dq_wei_unpacked.s67;
                weights_idx += 16 * 4;
                wei_local_idx += 16 * 4;
                */

                // [Packing weight]
                char4 wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
                char_slm_weight[wei_local_idx] = as_int(wei_1);
                char4 wei_2 = {dq_wei_unpacked.s45, dq_wei_unpacked.s67};
                char_slm_weight[wei_local_idx+1] = as_int(wei_2);
                weights_idx += 16 * 4;
                wei_local_idx += 16 * 2;
                // [Weight along local_id]
                // wei_local_idx += 8 * 2;
            }
            #endif

            wei_local_idx = sglid;

            barrier(CLK_LOCAL_MEM_FENCE);
        #endif  // USE_SLM : Restore compressed weight

        #if 0
        // wei_local_idx = sglid * 4;
        // [Packing weight]
        wei_local_idx = sglid * 2;
        // [Weight along local_id]
        // wei_local_idx = sglid * 2 * 8;
        __attribute__((opencl_unroll_hint)) for (uint ki = 0; ki < (2 * 16) / 4; ++ki) {
            char4 input_val = as_char4(_sub_group_shuffle(packed_in_0[0], ki));
            // char4 first_weight = ((char4 *)(&char_slm_weight[wei_local_idx]))[0];
            // char4 second_weight = ((char4 *)(&char_slm_weight[wei_local_idx+2]))[0];
            // [Packing weight]
            char4 first_weight = as_char4(((__local int *)(&char_slm_weight[wei_local_idx]))[0]);
            char4 second_weight = as_char4(((__local int *)(&char_slm_weight[wei_local_idx+1]))[0]);
            __attribute__((opencl_unroll_hint)) for (uint bi = 0; bi < 8; ++bi) {
                // [TEMP]
                // ((int *)(&acc_tmp[bi]))[0] = imad_SW(((int *)(&acc_tmp[bi]))[0],
                //                                         as_char4(_sub_group_shuffle(packed_in_0[bi % 4], (bi / 4) * 8 + ki)),
                //                                         ((char4 *)(&char_slm_weight[wei_local_idx]))[0]);
                // ((int *)(&acc_tmp[bi]))[1] = imad_SW(((int *)(&acc_tmp[bi]))[1],
                //                                         as_char4(_sub_group_shuffle(packed_in_0[bi % 4], (bi / 4) * 8 + ki)),
                //                                         ((char4 *)(&char_slm_weight[wei_local_idx+2]))[0]);

                // Chaged order
                // ((int *)(&acc_tmp[0]))[bi] = imad_SW(((int *)(&acc_tmp[0]))[bi],
                //                                         as_char4(_sub_group_shuffle(packed_in_0[bi / 2], (bi % 2) * 8 + ki)),
                //                                         ((char4 *)(&char_slm_weight[wei_local_idx]))[0]);
                // ((int *)(&acc_tmp[1]))[bi] = imad_SW(((int *)(&acc_tmp[1]))[bi],
                //                                         as_char4(_sub_group_shuffle(packed_in_0[bi / 2], (bi % 2) * 8 + ki)),
                //                                         ((char4 *)(&char_slm_weight[wei_local_idx+2]))[0]);

                // SW pipeline
                acc_tmp[0][bi] = imad_SW(acc_tmp[0][bi], input_val, first_weight);
                acc_tmp[1][bi] = imad_SW(acc_tmp[1][bi], input_val, second_weight);
                input_val = as_char4(_sub_group_shuffle(packed_in_0[(bi+1) / 2], ((bi+1) % 2) * 8 + ki));

                // if (get_group_id(0) == 0 && get_group_id(0) == 0 && bi == 0 && sglid == 0) {
                //     printf(">>> %p , %p \n", &(((char4 *)(&wei))[0]), &(((char4 *)(&wei))[1]));
                //     printf("  --- [%d %d %d %d] [%d %d %d %d]\n", (int)wei[4], (int)wei[5], (int)wei[6], (int)wei[7], (int)char_slm_weight[wei_local_idx + 2][0],
                //             (int)char_slm_weight[wei_local_idx + 2][1], (int)char_slm_weight[wei_local_idx + 3][0], (int)char_slm_weight[wei_local_idx + 3][1]);
                // }
            }

            // wei_local_idx += 16 * 4;
            // [Packing weight]
            wei_local_idx += 16 * 2;
            // [Weight along local_id]
            // wei_local_idx += 2;
        }  // ki < (TILE_IFM * SIMD) / TILE_K
        #else
        wei_local_idx = sglid * 2;
        for (uint ki = 0; ki < (2 * 16) / 4; ++ki) {
            char4 input_val = as_char4(_sub_group_shuffle(packed_in_0[0], ki));
            char4 first_weight = as_char4(((__local int *)(&char_slm_weight[wei_local_idx]))[0]);
            char4 second_weight = as_char4(((__local int *)(&char_slm_weight[wei_local_idx+1]))[0]);
            __attribute__((opencl_unroll_hint)) for (uint bi = 0; bi < 8; ++bi) {
                acc_tmp[0][bi] = imad_SW(acc_tmp[0][bi], input_val, first_weight);
                acc_tmp[1][bi] = imad_SW(acc_tmp[1][bi], input_val, second_weight);
                input_val = as_char4(_sub_group_shuffle(packed_in_0[(bi+1) / 2], ((bi+1) % 2) * 8 + ki));
            }
            wei_local_idx += 16 * 2;
        }
        #endif
        weights_offset += 4 * 16 * (8);

        // Get accumulated value
        #if 0
        for (uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K ; ++ki) {
            // Load compressed weight
            #if COMPRESSED_WEIGHTS_INT4
                #if USE_SLM
                    DQ_FILTER_VEC_TYPE wei = 0;
                    #define LOAD_FROM_SLM(vec2) vec2 = char_slm_weight[wei_local_idx]; wei_local_idx += SIMD;
                    #if TILE_K == 1
                        LOAD_FROM_SLM(wei.s01);
                    #elif TILE_K == 2
                        LOAD_FROM_SLM(wei.s01);
                        LOAD_FROM_SLM(wei.s23);
                    #elif TILE_K == 4
                        LOAD_FROM_SLM(wei.s01);
                        LOAD_FROM_SLM(wei.s23);
                        LOAD_FROM_SLM(wei.s45);
                        LOAD_FROM_SLM(wei.s67);
                    #else
                    #error "FC bf_tiled kernel: unsupported TILE_K size for SLM kernel"
                    #endif
                    #undef LOAD_FROM_SLM
                #else
                    FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                    #if DECOMPRESSION_SCALE_POST_OP
                        wei = UNPACK_MIXED_INT4x2(DQ_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
                    #else
                        wei = UNPACK_INT4x2(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
                    #endif
                #endif
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            // Calculate dzp and ds : COMPRESSED_WEIGHTS == 1 && USE_SLM == 1
            #if COMPRESSED_WEIGHTS && !USE_SLM
                #if DECOMPRESSION_SCALE_POST_OP
                    DQ_TYPE* dq_w = (DQ_TYPE*)(&wei);
                #else
                    ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                #endif

                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        // const uint w_idx = kii * TILE_OFM + fi;
                        const uint w_idx = fi * TILE_K + kii;
                        const uint offset_ofm = out_f + fi*SIMD + sglid;
                        #if !DECOMPRESSION_SCALE_POST_OP
                            // Apply scales before FMA to avoid FP16 overflow in case of INT8
                            #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                                const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH  +
                                                        ((kii + ki*TILE_K + ni*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                                ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                            #else
                                ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE ds = ACCUMULATOR_VAL_ONE;
                        #endif

                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                    ((kii + ki*TILE_K + ni*TILE_IFM*SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif

                        #if DECOMPRESSION_SCALE_POST_OP
                            dq_w[w_idx] = (dq_w[w_idx] - TO_DQ_TYPE(dzp));
                        #else
                            w[w_idx] = (w[w_idx] - dzp) * ds;
                        #endif
                    }
                }
            #endif
            weights_offset += TILE_K_OFM_PACKED * SIMD;

            unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                ((int*)(&acc_tmp[bi]))[0] = IMAD(((int*)(&acc_tmp[bi]))[0],
                            AS_DQ_TYPE_4(_sub_group_shuffle(packed_in_0[bi%4], (bi/4)*8 + ki)),
                            ((MAKE_VECTOR_TYPE(DQ_TYPE, 4) *)(&wei))[0]);
                ((int*)(&acc_tmp[bi]))[1] = IMAD(((int*)(&acc_tmp[bi]))[1],
                            AS_DQ_TYPE_4(_sub_group_shuffle(packed_in_0[bi%4], (bi/4)*8 + ki)),
                            ((MAKE_VECTOR_TYPE(DQ_TYPE, 4) *)(&wei))[1]);
            }

        #if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD > DECOMPRESSION_SCALE_GROUP_SIZE)
            unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi*SIMD + sglid;

                    #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                        const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                ((ni*TILE_IFM*SIMD + ki*TILE_K) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                        ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                    #else
                        ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                    #endif

                    ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += TO_ACCUMULATOR_TYPE(((int*)(&acc_tmp[bi]))[fi]) * ds * de_quantize_scale[bi%4];
                    acc_tmp[bi][fi] = 0;
                }
            }
        #endif
        }  // ki < (TILE_IFM * SIMD) / TILE_K
        #endif

#if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD <= DECOMPRESSION_SCALE_GROUP_SIZE)
        const uint ni_offset = ((ni*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
        unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                const uint offset_ofm = out_f + fi*SIMD + sglid;

                #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                    const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH + ni_offset;
                    ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                #else
                    ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                #endif
                ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += TO_ACCUMULATOR_TYPE(((int*)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi/2];
            }
        }
#endif
    }  // ni

    // =====================================================================================================================================
    // Leftovers
#if 0
#if MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    // Handle leftovers in normal case without alignment correction.
    #define LEFTOVER_IFM               (MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD))
    {
        #define LOAD_IN_0(bi) do {                                  \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                input_offset += TILE_IN_B_PITCH;                    \
            } while (false)

        CONST_LOOP(TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * TILE_B;
        unroll_for(uint ki = 0; ki < CEIL_DIV(LEFTOVER_IFM, TILE_K); ++ki) {
            #if USE_SLM
                FILTER_VEC_TYPE wei = 0;
            #endif

            #if COMPRESSED_WEIGHTS_INT4
                FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                wei = UNPACK_INT4x2(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            #if COMPRESSED_WEIGHTS
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        const uint w_idx = kii * TILE_OFM + fi;
                        uint offset_ofm = out_f + fi*SIMD + get_sub_group_local_id();
                        #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                            const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                      ((kii + ki*TILE_K + iterations*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                            ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                        #else
                            ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                        #endif

                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                    ((kii + ki*TILE_K + iterations*TILE_IFM*SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif
                        w[w_idx] = (w[w_idx] - dzp) * ds;
                    }
                }
            #endif
            weights_offset += TILE_K_OFM_PACKED * SIMD;

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                        const uint total_k = ki * TILE_K + kii;
                        if (total_k < LEFTOVER_IFM) {
                            INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                            ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[kii * TILE_OFM + fi];
                        }
                    }
                }
            }
        }
    }
    #undef LEFTOVER_IFM

#endif // MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
#endif

    // =====================================================================================================================================
    // Post-processing: bias, activation, fused-ops
    const uint CURRENT_TILE_B = TILE_B;
    // const uint CURRENT_TILE_B = TILE_B / 2;
    ACTIVATION_VEC_TYPE activated[CURRENT_TILE_B] = { };
    for (uint bi = 0; bi < CURRENT_TILE_B; ++bi) {
        activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
    }

#if BIAS_TERM
    #if TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0
        BIAS_VEC_TYPE bias = BIAS_BLOCK_READ(biases, out_f);
    #else
        BIAS_VEC_TYPE bias = 0;
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            ((BIAS_TYPE*)(&bias))[fi] = biases[out_f + sglid + fi * SIMD];
        }
    #endif
    unroll_for (uint bi = 0; bi < CURRENT_TILE_B; ++bi) {
        activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
    }
#endif

    OUTPUT_VEC_TYPE result[CURRENT_TILE_B] = { };
#if HAS_FUSED_OPS
    unroll_for (uint bi = 0; bi < CURRENT_TILE_B; ++bi) {
    #if TILE_OFM > 1
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            FUSED_OPS_VEC;
            result[bi][fi] = FUSED_OPS_RESULT_VEC;
        }
    #else
        FUSED_OPS_SCALAR;
        result[bi] = FUSED_OPS_RESULT_SCALAR;
    #endif // TILE_OFM > 1
    }
#else
    unroll_for (uint bi = 0; bi < CURRENT_TILE_B; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
    }
#endif

    // =====================================================================================================================================
    // Packing offset
    // uint output_batch_sglid = (sglid * TILE_K / 32);   // 0 to 1 : to batch direction
    // output_batch_sglid *= (TILE_OUT_B_PITCH * (TILE_B/2));
    // Write results
    uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;
    // uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET + output_batch_sglid;

    if (USE_BLOCK_WRITE && (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 || out_f + (TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
#if 0
#if IS_DYNAMIC
        #define WRITE_OUTPUT(bi) do {                                       \
                if (bi + out_b < BATCH_SIZE)                                \
                    OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);  \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#else
        #define WRITE_OUTPUT(bi) do {                                       \
                OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);      \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#endif
#endif

    #define WRITE_OUTPUT(bi) do {                                       \
            OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);      \
            output_offset += TILE_OUT_B_PITCH;                          \
        } while (false)
    CONST_LOOP(TILE_B, WRITE_OUTPUT);
    #undef WRITE_OUTPUT


    } else {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        output_offset += sglid;

        // TODO: Investigate why below code doesn't compile and check how it affects performance.
        //#define WRITE_OUTPUT_FEATURE(fi) do {                                                   \
        //        const bool should_write =                                                       \
        //            TILE_OUT_F_NUM %  (TILE_OFM * SIMD) == 0 ||                                 \
        //            out_f + (fi) * SIMD + sglid < TILE_OUT_F_NUM;                               \
        //        if (should_write) {                                                             \
        //            output[output_offset] = result[out_bi][fi];                                 \
        //        }                                                                               \
        //        output_offset += SIMD;                                                          \
        //    } while (false)
        //
        //#define WRITE_OUTPUT(bi) do {                                                           \
        //        const uint out_bi = bi;                                                         \
        //        CONST_LOOP(TILE_OFM, WRITE_OUTPUT_FEATURE);                                     \
        //        output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;                            \
        //    } while (false)
        //
        //CONST_LOOP(TILE_B, WRITE_OUTPUT);
        //#undef WRITE_OUTPUT
        //#undef WRITE_OUTPUT_FEATURE

        for (uint bi = 0; bi < TILE_B; ++bi) {
            for (uint fi = 0; fi < TILE_OFM; ++fi) {
                const bool should_write =
#if IS_DYNAMIC
                    bi + out_b < BATCH_SIZE &&
#endif
                    (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 ||
                    out_f + fi * SIMD + sglid < TILE_OUT_F_NUM);
                if (should_write) {
                    output[output_offset] = ((OUTPUT_TYPE*)(&result[bi]))[fi];
                }
                output_offset += SIMD;
            }
            output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
        }
    }
    // =====================================================================================================================================
}

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(fc)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if USE_SLM
    // [Packing weight]
    // __local DQ_TYPE dq_wei_local_mem[TILE_IFM * SIMD * TILE_OFM * SIMD];
    __local int dq_wei_local_mem[SIMD * TILE_OFM * SIMD];
#endif

    FUNC_CALL(fc_bf_tiled_kernel_default)(
        OPTIONAL_SHAPE_INFO_TENSOR
        input,
    #if DECOMPRESSION_SCALE_TERM
        decompression_scale,
    #endif
    #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
        decompression_zp,
    #endif
        output,
        weights
    #if USE_SLM
        , dq_wei_local_mem
    #endif
    #if BIAS_TERM
        , biases
    #endif
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_ARGS
    #endif
    );
}

#undef INPUT_VEC_TYPE
#undef ACCUMULATOR_VEC_TYPE
#undef FILTER_VEC_TYPE
#undef BIAS_VEC_TYPE
#undef OUTPUT_VEC_TYPE
#undef ACTIVATION_VEC_TYPE
#undef TO_OUTPUT_VEC_TYPE
#undef TO_ACTIVATION_VEC_TYPE

#undef INPUT_BLOCK_READ
#undef FILTER_BLOCK_READ
#undef BIAS_BLOCK_READ
#undef OUTPUT_BLOCK_WRITE

#undef USE_BLOCK_WRITE

#undef MAIN_LOOP_ELEMENTS_COUNT
