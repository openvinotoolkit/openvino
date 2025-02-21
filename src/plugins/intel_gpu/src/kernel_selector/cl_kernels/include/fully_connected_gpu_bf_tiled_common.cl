// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define FUNC_NAME_BATCHED CAT(fc_bf_tiled_kernel_forced_tile_b, FORCED_TILE_B)
#define FUNC_NAME CAT(_, CAT(CAT(FUNC_NAME_BATCHED, _), KERNEL_ID))

inline void (FUNC_NAME)(
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
    uint gid = (uint)get_group_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    // Dispatch as bs_fs_bsv_fsv, where bsv = DISPATCH_BSV and fsv = DISPATCH_FSV.
    // This allows more fine grained control over dispatch order than using work-groups and
    // avoids requirement of threads being available for whole work-group.
    // It could hovewer have some drawbacks like not providing physical locality or not using
    // full dispatch pipeline.
    uint feature_mini_block = gid % DISPATCH_FSV;
    uint batch_mini_block = gid / DISPATCH_FSV % DISPATCH_BSV;
    #ifdef SWIGLU_LENGTH
    uint feature_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV) % (CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);
    uint batch_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV * CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);
    #else
    uint feature_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV) % (CEIL_DIV(TILE_OUT_F_NUM, OUTER_OFM * TILE_OFM * SIMD) / DISPATCH_FSV);
    uint batch_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV * CEIL_DIV(TILE_OUT_F_NUM, OUTER_OFM * TILE_OFM * SIMD) / DISPATCH_FSV);
    #endif

    #ifdef SWIGLU_LENGTH
    uint out_f = (feature_mega_block * DISPATCH_FSV + feature_mini_block) * (TILE_OFM * SIMD);
    #else
    uint out_f = (feature_mega_block * DISPATCH_FSV + feature_mini_block) * (OUTER_OFM * TILE_OFM * SIMD);
    #endif
    uint out_b = ((batch_mega_block * DISPATCH_BSV + batch_mini_block) * FORCED_TILE_B);

    ACCUMULATOR_VEC_TYPE acc[FORCED_TILE_B] = { };
    INPUT_VEC_TYPE       in_0[FORCED_TILE_B] = { };

    FILTER_VEC_TYPE wei = 0;
#if OUTPUT_3D
    uint out_b0 = out_b / OUTPUT_FEATURE_NUM;
    uint out_b1 = out_b % OUTPUT_FEATURE_NUM;
    uint input_offset = out_b0 * INPUT0_BATCH_PITCH + out_b1 * INPUT0_FEATURE_PITCH + INPUT0_OFFSET;
#else
    uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
#endif

#if COMPRESSED_WEIGHTS_INT4
    uint weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2);
#else
    uint weights_offset = out_f * INPUT_ELEMENTS_COUNT;
#endif

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

    ACTIVATION_VEC_TYPE activated[FORCED_TILE_B] = { };
#if OUTER_OFM > 1
    uint input_offset_init = input_offset;
    uint weights_offset_init = weights_offset;
    uint out_f_init = out_f;
    unroll_for (uint oi = 0; oi < OUTER_OFM; ++oi) {
        input_offset = input_offset_init;
        #ifdef SWIGLU_LENGTH
        weights_offset = weights_offset_init + oi * (FILTER_IFM_NUM / (TILE_K_OFM / TILE_K_OFM_PACKED) ) * SWIGLU_LENGTH;
        out_f += SWIGLU_LENGTH * oi;
        #else
        out_f += TILE_OFM * SIMD * oi;
        #endif
#endif

#if REALIGN_FP16_OFFSET
    // For fp16 we need to ensure that all block reads are aligned to 4 byte (2 words) boundary.
    // To do this solve first input feature separately.
    {
        INPUT0_TYPE tmp_input = input[input_offset + get_sub_group_local_id() % FORCED_TILE_B * TILE_IN_B_PITCH];
        ACCUMULATOR_VEC_TYPE tmp_wei = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(FILTER_TYPE, TILE_OFM, weights, weights_offset));
        #if COMPRESSED_WEIGHTS
            tmp_wei = (tmp_wei - d_zp) * d_scale;
        #endif
        unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            acc[bi] = _sub_group_shuffle(tmp_input, bi) * tmp_wei;
        }

        weights_offset += TILE_OFM * SIMD;
        input_offset += 1;
    }
#endif
    // =====================================================================================================================================
    // Main computation loop
    uint iterations = MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD);
    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Load input.
        #define LOAD_IN_0(bi) do {                                  \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                input_offset += TILE_IN_B_PITCH;                    \
            } while (false)

        CONST_LOOP(FORCED_TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * FORCED_TILE_B;
        // NOTE: Manually unrolling multiplication loop leads to lower register pressure and allows for bigger block sizes,
        //       but significantly degrades readability and generality of code.
        //       It doesn't also show noticable performance improvement on tested configurations.
        ACCUMULATOR_VEC_TYPE acc_tmp[FORCED_TILE_B] = { };

        unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
            #if COMPRESSED_WEIGHTS_INT4
                FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            #if COMPRESSED_WEIGHTS
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
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
                        w[W_IDX] = (w[W_IDX] - dzp) * ds;
                    }
                }
            #endif
            weights_offset += TILE_K_OFM_PACKED * TILE_OFM_PER_OSV_SIZE * SIMD;

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                const uint total_k = ki * TILE_K + kii;
                unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
                    INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                    unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                        ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                    }
                }
            }
#if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD > DECOMPRESSION_SCALE_GROUP_SIZE)
            unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi*SIMD + sglid;

                    #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                        const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                ((ni*TILE_IFM*SIMD + ki*TILE_K) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                        ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                    #else
                        ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                    #endif
                    #if TILE_OFM > 1
                    ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] * ds;
                    acc_tmp[bi][fi] = 0;
                    #else
                    acc[bi] += acc_tmp[bi] * ds;
                    acc_tmp[bi] = 0;
                    #endif
                }
            }
#endif
        }
#if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD <= DECOMPRESSION_SCALE_GROUP_SIZE)
        unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                const uint offset_ofm = out_f + fi*SIMD + sglid;

                #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                    const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                ((ni*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                    ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                #else
                    ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                #endif
                #if TILE_OFM > 1
                ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] * ds;
                acc_tmp[bi][fi] = 0;
                #else
                acc[bi] += acc_tmp[bi] * ds;
                acc_tmp[bi] = 0;
                #endif
            }
        }
#endif

#if !DECOMPRESSION_SCALE_POST_OP
        unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi];
            }
        }
#endif


    }
    // =====================================================================================================================================
    // Leftovers
#if MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    // Handle leftovers in normal case without alignment correction.
    #define LEFTOVER_IFM               (MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD))
    {
        #define LOAD_IN_0(bi) do {                                  \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                input_offset += TILE_IN_B_PITCH;                    \
            } while (false)

        CONST_LOOP(FORCED_TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * FORCED_TILE_B;
        unroll_for(uint ki = 0; ki < CEIL_DIV(LEFTOVER_IFM, TILE_K); ++ki) {
            #if COMPRESSED_WEIGHTS_INT4
                FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            #if COMPRESSED_WEIGHTS
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
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
                        w[W_IDX] = (w[W_IDX] - dzp) * ds;
                    }
                }
            #endif
            weights_offset += TILE_K_OFM_PACKED * TILE_OFM_PER_OSV_SIZE * SIMD;

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                    unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
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
    // =====================================================================================================================================
    // Post-processing: bias, activation, fused-ops
    for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
        #ifdef SWIGLU_LENGTH
        #if SWIGLU_SPLIT_TO_GLU_IDX == 0
        if (oi == 0) {
            activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
            activated[bi] /= (ACCUMULATOR_VAL_ONE + native_exp(-(ACCUMULATOR_VAL_ONE * activated[bi])));
        } else {
            activated[bi] *= TO_ACTIVATION_VEC_TYPE(acc[bi]);
        }
        #else
        if (oi == 0) {
            // swish
            activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
        } else {
            acc[bi] /= (ACCUMULATOR_VAL_ONE + native_exp(-(ACCUMULATOR_VAL_ONE * acc[bi])));
            activated[bi] *= TO_ACTIVATION_VEC_TYPE(acc[bi]);
        }
        #endif
        #else
        activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
        #endif
#if OUTER_OFM > 1
        acc[bi] = 0;
#endif
    }

#if OUTER_OFM > 1 && defined(SWIGLU_LENGTH)
    }
    out_f = out_f_init;
#endif

#if BIAS_TERM
    #if TILE_OUT_F_NUM % (OUTER_OFM * TILE_OFM * SIMD) == 0
        BIAS_VEC_TYPE bias = BIAS_BLOCK_READ(biases, out_f);
    #else
        BIAS_VEC_TYPE bias = 0;
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            ((BIAS_TYPE*)(&bias))[fi] = biases[out_f + sglid + fi * SIMD];
        }
    #endif
    unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
        activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
    }
#endif

    OUTPUT_VEC_TYPE result[FORCED_TILE_B] = { };
#if HAS_FUSED_OPS
    unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
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
    unroll_for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
    }
#endif
    // =====================================================================================================================================
    // Write results
    uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;

    if (USE_BLOCK_WRITE && (TILE_OUT_F_NUM % (OUTER_OFM * TILE_OFM * SIMD) == 0 || out_f + (OUTER_OFM * TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
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
        CONST_LOOP(FORCED_TILE_B, WRITE_OUTPUT);
        #undef WRITE_OUTPUT
    } else {
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
        //CONST_LOOP(FORCED_TILE_B, WRITE_OUTPUT);
        //#undef WRITE_OUTPUT
        //#undef WRITE_OUTPUT_FEATURE

        for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            for (uint fi = 0; fi < TILE_OFM; ++fi) {
                const bool should_write =
#if IS_DYNAMIC
                    bi + out_b < BATCH_SIZE &&
#endif
                    (TILE_OUT_F_NUM % (OUTER_OFM * TILE_OFM * SIMD) == 0 ||
                    out_f + fi * SIMD + sglid < TILE_OUT_F_NUM);
                if (should_write) {
                    output[output_offset] = ((OUTPUT_TYPE*)(&result[bi]))[fi];
                }
                output_offset += SIMD;
            }
            output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
        }
    }
#if OUTER_OFM > 1 && !defined(SWIGLU_LENGTH)
    }
#endif
    // =====================================================================================================================================
}

#undef FUNC_NAME_BATCHED
#undef FUNC_NAME
