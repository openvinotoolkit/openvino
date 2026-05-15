// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core computation function for fully_connected_gpu_bf_tiled_dyn_b.
// Included multiple times with different FORCED_TILE_B values (1..8).
// Each variant processes exactly FORCED_TILE_B batch elements per work-item.
// The caller (entry point) computes out_b and out_f and passes them explicitly,
// enabling main+tail two-phase dispatch for non-exact batch tiling.

#define DYN_B_FUNC_BATCHED  CAT(fc_dyn_b_tile, FORCED_TILE_B)
#define DYN_B_FUNC_NAME     CAT(_, CAT(CAT(DYN_B_FUNC_BATCHED, _), KERNEL_ID))

inline void (DYN_B_FUNC_NAME)(
    uint out_b,
    uint out_f,
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
    // ---- Work-item identification ----
    const uint sglid = (uint)get_sub_group_local_id();
    // out_b and out_f are passed from the dispatcher (main or tail phase)

    // ---- Accumulators ----
    ACCUMULATOR_VEC_TYPE acc[FORCED_TILE_B]  = { };
    INPUT_VEC_TYPE       in_0[FORCED_TILE_B] = { };
    FILTER_VEC_TYPE      wei = 0;

    // ---- Input offset ----
#if OUTPUT_3D
    const uint out_b0 = out_b / OUTPUT_FEATURE_NUM;
    const uint out_b1 = out_b % OUTPUT_FEATURE_NUM;
    uint input_offset = out_b0 * INPUT0_BATCH_PITCH + out_b1 * INPUT0_FEATURE_PITCH + INPUT0_OFFSET;
#else
    uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
#endif

    // ---- Weight offset (INT4 packed) ----
#if TILE_OFM == 1 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
    const uint osv32_weight_base = ((out_f >> 5) << 5);
    const uint osv_weight_stride = (INPUT_ELEMENTS_COUNT >> 1);
    const uint out_f_offset      = ((out_f >> 4) & 0x1) << 4;
    uint weights_offset = osv32_weight_base * osv_weight_stride + out_f_offset;
#elif TILE_OFM == 2 && FILTER_LAYOUT_OS_IS_YX_OSV64_ISV2
    const uint osv64_weight_base = ((out_f >> 6) << 6);
    const uint osv_weight_stride = (INPUT_ELEMENTS_COUNT >> 1);
    const uint out_f_offset      = ((out_f >> 5) & 0x1) << 5;
    uint weights_offset = osv64_weight_base * osv_weight_stride + out_f_offset;
#else
    uint weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2);
#endif

    // ---- Pre-load per-tensor scale/zp (if scale_groups == 1) ----
#if COMPRESSED_WEIGHTS && DECOMPRESSION_SCALE_GROUPS_NUM == 1
    #if DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_scale = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_SCALE_TYPE, TILE_OFM, decompression_scale, out_f));
    #elif DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_scale = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of * SIMD + sglid;
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
            uint offset = out_f + of * SIMD + sglid;
            if (offset < DECOMPRESSION_ZP_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_zp))[of] = decompression_zp[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_zp = decompression_zp[0];
    #endif
    ACCUMULATOR_TYPE* d_zps = (ACCUMULATOR_TYPE*)(&d_zp);
#endif

    // ---- FP16 offset realignment ----
#if REALIGN_FP16_OFFSET
    {
        INPUT0_TYPE tmp_input = input[input_offset + sglid % FORCED_TILE_B * TILE_IN_B_PITCH];
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

    // ================================================================
    // Main computation loop
    // ================================================================
    const uint iterations = MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD);
    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Load input tile: FORCED_TILE_B elements (no bounds check needed)
        unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            in_0[bi] = INPUT_BLOCK_READ(input, input_offset);
            input_offset += TILE_IN_B_PITCH;
        }
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * FORCED_TILE_B;

#if DECOMPRESSION_SCALE_POST_OP
        ACCUMULATOR_VEC_TYPE acc_tmp[FORCED_TILE_B] = { };
#endif

        // Inner K loop: load/unpack INT4 weights, decompress, multiply-accumulate
        unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
            FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
            wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));

            ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
            unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi * SIMD + sglid;

                    // Decompression scale
                    #if !DECOMPRESSION_SCALE_POST_OP
                        #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                            const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                     ((kii + ki * TILE_K + ni * TILE_IFM * SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE) * DECOMPRESSION_SCALE_FEATURE_PITCH;
                            ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                        #else
                            ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                        #endif
                    #else
                        ACCUMULATOR_TYPE ds = ACCUMULATOR_VAL_ONE;
                    #endif

                    // Zero point
                    #if DECOMPRESSION_ZP_TERM
                        #if DECOMPRESSION_ZP_SCALAR
                            ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                        #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                            const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                  ((kii + ki * TILE_K + ni * TILE_IFM * SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
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

            // Multiply-accumulate via sub-group shuffle broadcast
            unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                const uint total_k = ki * TILE_K + kii;
                unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
                    INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
#if DECOMPRESSION_SCALE_POST_OP
                        #if TILE_OFM > 1
                            ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                        #else
                            acc_tmp[bi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                        #endif
#else
                        #if TILE_OFM > 1
                            ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                        #else
                            acc[bi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                        #endif
#endif
                    }
                }
            }
            weights_offset += TILE_K_OFM_PACKED * TILE_OFM_PER_OSV_SIZE * SIMD;

            // Apply deferred scale mid-loop when scale group < TILE_IFM*SIMD
#if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD > DECOMPRESSION_SCALE_GROUP_SIZE)
            unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi * SIMD + sglid;
                    #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                        const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                 ((ni * TILE_IFM * SIMD + ki * TILE_K) / DECOMPRESSION_SCALE_GROUP_SIZE) * DECOMPRESSION_SCALE_FEATURE_PITCH;
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
        } // end ki loop

        // Apply deferred scale at end of outer loop iteration
#if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD <= DECOMPRESSION_SCALE_GROUP_SIZE)
        unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                const uint offset_ofm = out_f + fi * SIMD + sglid;
                #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                    const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                             ((ni * TILE_IFM * SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE) * DECOMPRESSION_SCALE_FEATURE_PITCH;
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
    } // end ni loop

    // ================================================================
    // Leftovers: remaining IFM elements not divisible by (TILE_IFM * SIMD)
    // ================================================================
#if MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    #define LEFTOVER_IFM (MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD))
    {
        unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            in_0[bi] = INPUT_BLOCK_READ(input, input_offset);
            input_offset += TILE_IN_B_PITCH;
        }
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * FORCED_TILE_B;

        unroll_for(uint ki = 0; ki < CEIL_DIV(LEFTOVER_IFM, TILE_K); ++ki) {
            FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
            wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));

            ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
            unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi * SIMD + sglid;
                    #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                        const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                 ((kii + ki * TILE_K + iterations * TILE_IFM * SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE) * DECOMPRESSION_SCALE_FEATURE_PITCH;
                        ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                    #else
                        ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                    #endif

                    #if DECOMPRESSION_ZP_TERM
                        #if DECOMPRESSION_ZP_SCALAR
                            ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                        #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                            const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                  ((kii + ki * TILE_K + iterations * TILE_IFM * SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
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
            weights_offset += TILE_K_OFM_PACKED * TILE_OFM_PER_OSV_SIZE * SIMD;

            unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
                        const uint total_k = ki * TILE_K + kii;
                        if (total_k < LEFTOVER_IFM) {
                            INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                            #if TILE_OFM > 1
                                ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                            #else
                                acc[bi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                            #endif
                        }
                    }
                }
            }
        }
    }
    #undef LEFTOVER_IFM
#endif

    // ================================================================
    // Post-processing: bias, activation, fused ops
    // ================================================================
    ACTIVATION_VEC_TYPE activated[FORCED_TILE_B] = { };
    unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
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
    unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
        activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
    }
#endif

    OUTPUT_VEC_TYPE result[FORCED_TILE_B] = { };
#if HAS_FUSED_OPS
    unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
    #if TILE_OFM > 1
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            FUSED_OPS_VEC;
            result[bi][fi] = FUSED_OPS_RESULT_VEC;
        }
    #else
        FUSED_OPS_SCALAR;
        result[bi] = FUSED_OPS_RESULT_SCALAR;
    #endif
    }
#else
    unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
    }
#endif

    // ================================================================
    // Write results (tile variant matches actual element count -- no bounds check)
    // ================================================================
    uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;

    if (USE_BLOCK_WRITE && (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 || out_f + (TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
        unroll_for(uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);
            output_offset += TILE_OUT_B_PITCH;
        }
    } else {
        output_offset += sglid;
        for (uint bi = 0; bi < FORCED_TILE_B; ++bi) {
            for (uint fi = 0; fi < TILE_OFM; ++fi) {
                if (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 || out_f + fi * SIMD + sglid < TILE_OUT_F_NUM) {
                    output[output_offset] = ((OUTPUT_TYPE*)(&result[bi]))[fi];
                }
                output_offset += SIMD;
            }
            output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
        }
    }
}

#undef DYN_B_FUNC_BATCHED
#undef DYN_B_FUNC_NAME
