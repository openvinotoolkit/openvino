// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/imad.cl"
#if QUANTIZATION_TERM
#    define ACCUMULATOR_TYPE int
#    define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#    define ACTIVATION_TYPE float
#    define TO_ACTIVATION_TYPE(x) convert_float(x)
#else
#    define ACCUMULATOR_TYPE INPUT0_TYPE
#    define TO_ACCUMULATOR_TYPE(x) TO_INPUT0_TYPE(x)
#    define ACTIVATION_TYPE INPUT0_TYPE
#    define TO_ACTIVATION_TYPE(x) TO_INPUT0_TYPE(x)
#endif

#if NON_BLOCK_LOAD != 1
// block loads for inputs and weights should be fastest, but compiler seems
// to do better with a mix, regular loads for inputs and block loads for weights.
#define BLOCK_LOAD_WEIGHTS
#endif
// Input reading operation is always blocked.
#define BLOCK_LOAD_INPUTS

// need KERNEL width for first output + STRIDE more for each additional.
#define IN_BLOCK_WIDTH  ((FILTER_SIZE_X - 1) * DILATION_SIZE_X + STRIDE_SIZE_X * (OUT_BLOCK_WIDTH  - 1) + 1)
#define IN_BLOCK_HEIGHT ((FILTER_SIZE_Y - 1) * DILATION_SIZE_Y + STRIDE_SIZE_Y * (OUT_BLOCK_HEIGHT - 1) + 1)

// for imad we are packing 4 8bit activations per 32 bit SIMD lane
// if we later add 4bit, then PACK would be 8.
#define PACK 4

#define TYPE_N_(type, n) type##n
#define TYPE_N(type, n) TYPE_N_(type, n)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define INPUT0_TYPE_4 TYPE_N(INPUT0_TYPE, 4)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)
#define AS_FILTER_TYPE_4(x) AS_TYPE_N(FILTER_TYPE, 4, x)

#if INPUT0_PAD_BEFORE_SIZE_X != 0 || INPUT0_PAD_BEFORE_SIZE_Y != 0
    #define NON_ZERO_INPUT0_PAD_BEFORE
#endif

#if !defined COMPENSATION_TERM || \
    (defined COMPENSATION_TERM && defined NON_ZERO_INPUT0_PAD_BEFORE)
    #define SHOULD_BALANCE_COMPENSATION
#endif

#if defined ASYMMETRIC_DATA_QUANTIZATION && defined SHOULD_BALANCE_COMPENSATION
    #define SHOULD_USE_DATA_ZP
#endif

#if defined ASYMMETRIC_DATA_QUANTIZATION && \
    defined ASYMMETRIC_WEIGHTS_QUANTIZATION && \
    defined SHOULD_BALANCE_COMPENSATION
    #define SHOULD_USE_DATA_AND_WEIGHTS_ZP
#endif

#ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
    #define FILTER_TYPE_4 TYPE_N(FILTER_TYPE, 4)
#endif

// int8 conv_input and weights data is packed to int32 "batches",
// int/uint pointers here instead of INPUT0_TYPE/FILTER_TYPE for convenience
REQD_SUB_GROUP_SIZE(SIMD_SIZE)
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
KERNEL (fused_convolution_eltwise_gpu_imad)(
#if INPUT0_LAYOUT_B_FS_YX_FSV16
    const __global INPUT0_TYPE* conv_input,
#else
    const __global PACKED_TYPE   *conv_input,
#endif
    __global OUTPUT_TYPE         *restrict output,
    const __global int           *weights
#if BIAS_TERM
    , const __global BIAS_TYPE     *biases
#endif
#ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
    , const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp
#endif
#ifdef ASYMMETRIC_DATA_QUANTIZATION
    , const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp
#endif
#ifdef COMPENSATION_TERM
    , const __global COMPENSATION_TYPE *compensation
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint oc = (uint)get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column
    const uint or = (uint)get_global_id(1) * OUT_BLOCK_HEIGHT; // or = Output Row
    const uint fm = get_global_id(2);                          // fm = Feature Map = od = Output Depth, SIMD is across this dimension, WG is 1x1x16
    const uint fmg = get_group_id(2);
    const uint lid = get_local_id(2);
    const uint batch = fm / (ALIGN(FILTER_OFM_NUM, SIMD_SIZE) * FILTER_GROUPS_NUM);
#if GROUPED
    const uint g = (fm / ALIGN(FILTER_OFM_NUM, SIMD_SIZE) % FILTER_GROUPS_NUM);
    const uint ofmg = fmg % CEIL_DIV(FILTER_OFM_NUM, SIMD_SIZE);
#else
    const uint g = 0;
    const uint ofmg = (fmg % (_OD  / SIMD_SIZE));
#endif
    const uint f = fm % ALIGN(FILTER_OFM_NUM, SIMD_SIZE) + g * FILTER_OFM_NUM;
    const uint sglid = get_sub_group_local_id();

    const int input_x = oc * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = or * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    PACKED_TYPE in[IN_BLOCK_HEIGHT];
    #ifdef SHOULD_USE_DATA_ZP
        #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % PACK != 0))
            uint data_zp_idx = g * FILTER_IFM_NUM;
        #else
            uint data_zp_idx = (g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * PACK;
        #endif
        PACKED_TYPE data_zp_val;
    #endif
    ACCUMULATOR_TYPE out[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT] = { 0 };  // this is the 32 bit signed accumulator that must be converted to 8 bits before final write.

    #define NUM_FILTERS (FILTER_SIZE_Y * FILTER_SIZE_X)
    int w[NUM_FILTERS];

    #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
        int weights_zp_val = as_int((FILTER_TYPE_4)weights_zp[f]);
        #if FILTER_IFM_NUM % PACK != 0
            int weights_zp_vec_partial;
            weights_zp_vec_partial = weights_zp_val;
            FILTER_TYPE* wzp_p = (FILTER_TYPE*)&weights_zp_vec_partial;
            unroll_for (uint in_f = FILTER_IFM_NUM % PACK; in_f < PACK; in_f++) {
                wzp_p[in_f] = 0;
            }
        #endif
    #endif

    int in_addr;

#if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % PACK != 0))
    int in_start_addr = INPUT0_GET_INDEX(batch, 0, input_y, input_x + sglid);
#endif

#ifdef BLOCK_LOAD_WEIGHTS
    int weight_addr = (ofmg * CEIL_DIV(FILTER_IFM_NUM, PACK) * FILTER_SIZE_Y * FILTER_SIZE_X * SIMD_SIZE) + (g * FILTER_GROUPS_PITCH / 4);
#else
    int weight_addr = (ofmg * CEIL_DIV(FILTER_IFM_NUM, PACK) * FILTER_SIZE_Y * FILTER_SIZE_X * SIMD_SIZE) + (g * FILTER_GROUPS_PITCH / 4) + sglid;
#endif
    uint input_size = (_ID * (INPUT0_SIZE_Y + IHPAD) * (INPUT0_SIZE_X + IWPAD)) / PACK; // dividing by PACK to get right number of 32bit entities.

    // For imad we do 4X less input feature map iterations since we are packing 4 of them in each uchar4.
    __attribute__((opencl_unroll_hint(1)))
    for(int kd = 0; kd < CEIL_DIV(FILTER_IFM_NUM, PACK); kd++)
    {
        #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
            #if FILTER_IFM_NUM % PACK != 0
                if ((kd + 1) * PACK >= ALIGN(FILTER_IFM_NUM, PACK)) {
                    weights_zp_val = weights_zp_vec_partial;
                }
            #endif
        #endif
        #if INPUT0_LAYOUT_B_FS_YX_FSV16
            #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % PACK != 0))
                int feature_location = kd * PACK + g * FILTER_IFM_NUM;
            #else
                in_addr = INPUT0_GET_INDEX(batch, (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * PACK, input_y, input_x + sglid);
            #endif
        #else
            #ifdef BLOCK_LOAD_INPUTS
                in_addr = INPUT0_OFFSET + (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * INPUT0_FEATURE_PITCH + input_y * INPUT0_Y_PITCH + input_x;
            #else
                in_addr = INPUT0_OFFSET + (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * INPUT0_FEATURE_PITCH + input_y * INPUT0_Y_PITCH + input_x + sglid;
            #endif
            in_addr += batch * input_size;  // adjust for batching
        #endif
        #ifdef SHOULD_USE_DATA_ZP
            #if INPUT0_LAYOUT_B_FS_YX_FSV16
                #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % PACK != 0))
                    INPUT0_TYPE* input_zp_int8_arr = (INPUT0_TYPE*) &data_zp_val;
                    for (uint v = 0; v < PACK; v++) {
                        input_zp_int8_arr[v] = activations_zp[feature_location + v];
                    }
                #else
                    data_zp_val = *(__global PACKED_TYPE*)(activations_zp + (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * PACK);
                #endif
            #else
                data_zp_val = AS_PACKED_TYPE(*((__global PACKED_TYPE*)activations_zp + (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK))));
            #endif
        #endif

        #ifdef SHOULD_USE_DATA_AND_WEIGHTS_ZP
            ACCUMULATOR_TYPE dotProdAZPxWZP;
            dotProdAZPxWZP = 0;
            dotProdAZPxWZP = TO_ACCUMULATOR_TYPE(IMAD(dotProdAZPxWZP, AS_INPUT0_TYPE_4(data_zp_val), AS_FILTER_TYPE_4(weights_zp_val)));
        #endif

        for(uint reg = 0; reg < IN_BLOCK_HEIGHT; reg++) {
            #ifdef SHOULD_USE_DATA_ZP
                const uint x_idx = input_x + sglid;
                const uint y_idx = input_y + reg;

                const bool input_on_padding = (((x_idx < 0) || (x_idx >= INPUT0_SIZE_X)) ||
                                               ((y_idx < 0) || (y_idx >= INPUT0_SIZE_Y)));
            #endif

            #if INPUT0_LAYOUT_B_FS_YX_FSV16
                #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % PACK != 0))
                    #ifdef SHOULD_USE_DATA_ZP
                        if (input_on_padding) {
                            in[reg] = data_zp_val;
                        } else {
                    #endif
                    INPUT0_TYPE* input_int8_arr = (INPUT0_TYPE*) &in[reg];
                    in_addr = in_start_addr + reg * INPUT0_Y_PITCH * FSV;
                    for (uint v = 0; v < PACK; v++) {
                        int f_addr = ((feature_location + v) / FSV + INPUT0_PAD_BEFORE_FEATURE_NUM / FSV) * \
                                      INPUT0_FEATURE_PITCH * FSV  + (feature_location + v) % FSV;
                        input_int8_arr[v] = conv_input[in_addr + f_addr];
                    }
                    #ifdef SHOULD_USE_DATA_ZP
                        }
                    #endif
                #else
                    #ifdef SHOULD_USE_DATA_ZP
                        if (input_on_padding)
                            in[reg] = data_zp_val;
                        else
                    #endif
                        in[reg] = *(__global PACKED_TYPE*)(conv_input + in_addr);
                        in_addr += (INPUT0_SIZE_X + IWPAD) * 16;
                 #endif
            #else
                #ifdef BLOCK_LOAD_INPUTS
                    in[reg] = AS_PACKED_TYPE(_sub_group_block_read((const __global uint*) &conv_input[in_addr]));
                    #ifdef SHOULD_USE_DATA_ZP
                        if (input_on_padding)
                            in[reg] = data_zp_val;
                    #endif
                #else
                    #ifdef SHOULD_USE_DATA_ZP
                        if (input_on_padding)
                            in[reg] = data_zp_val;
                        else
                    #endif
                            in[reg] = AS_PACKED_TYPE(conv_input[in_addr]); // read SIMD_SIZE elements wide
                #endif
                in_addr += (INPUT0_SIZE_X + IWPAD); // move to next row down
            #endif
        }

        #ifdef BLOCK_LOAD_WEIGHTS
            *((int8*)&w[0]) = as_int8(_sub_group_block_read8((const __global uint*) &weights[weight_addr]));
            w[8] = as_int(_sub_group_block_read((const __global uint*) &weights[weight_addr + (SIMD_SIZE<<3)]));
            weight_addr += SIMD_SIZE*NUM_FILTERS;
        #else
            for(int pf = 0; pf < NUM_FILTERS; pf++) {
                w[pf] = weights[weight_addr];
                weight_addr += SIMD_SIZE;
            }
        #endif

        int wi = 0;
        // This loop is temporarily not unrolled because the unroll causes TeamCity hangs.
        //__attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
        for (int kr = 0; kr < FILTER_SIZE_Y; ++kr) // kr = Kernel Row
        {
            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (int kc = 0; kc < FILTER_SIZE_X; ++kc) // kc = Kernel Column
            {
                #ifdef SHOULD_USE_DATA_ZP
                    ACCUMULATOR_TYPE dotProdAZPxW = 0;
                    dotProdAZPxW = TO_ACCUMULATOR_TYPE(IMAD(dotProdAZPxW, AS_INPUT0_TYPE_4(data_zp_val), AS_FILTER_TYPE_4(w[wi])));
                #endif

                unroll_for (int br = 0; br < OUT_BLOCK_HEIGHT; br++) {
                    unroll_for (int bc = 0; bc < OUT_BLOCK_WIDTH; bc++) {
                        INPUT0_TYPE_4 inputs = AS_INPUT0_TYPE_4(sub_group_broadcast(in[br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y],
                                                                                    bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X));

                        out[br * OUT_BLOCK_WIDTH + bc] = TO_ACCUMULATOR_TYPE(IMAD(out[br * OUT_BLOCK_WIDTH + bc], inputs, AS_FILTER_TYPE_4(w[wi])));

                        #if !defined COMPENSATION_TERM && defined ASYMMETRIC_DATA_QUANTIZATION
                            out[br * OUT_BLOCK_WIDTH + bc] -= dotProdAZPxW;
                        #endif

                        #if (!defined COMPENSATION_TERM && \
                                defined ASYMMETRIC_DATA_QUANTIZATION && \
                                defined ASYMMETRIC_WEIGHTS_QUANTIZATION)
                            out[br * OUT_BLOCK_WIDTH + bc] += dotProdAZPxWZP;
                        #endif

                        #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
                            ACCUMULATOR_TYPE dotProdAxWZP = 0;
                            dotProdAxWZP = TO_ACCUMULATOR_TYPE(IMAD(dotProdAxWZP, inputs, AS_FILTER_TYPE_4(weights_zp_val)));
                            out[br * OUT_BLOCK_WIDTH + bc] -= dotProdAxWZP;
                        #endif
                    }
                }
                wi++;
            }
        }
    } //for kd

    // Compiler emits worse code when GET_DATA_B_FS_YX_FSV4_INDEX is called inside the loop
    // to calculate out_idx and eltw_idx. Calculate offsets with GET_DATA_B_FS_YX_FSV4_INDEX before
    // entering the loop, and have a simple expressions for indexes inside the loop.
    const uint output_idx_offset = GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, batch, f, or, oc);
    const uint output_row_size_bytes = (OUTPUT_SIZE_X + OWPAD) * PACK;

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD;
#endif

    #ifdef COMPENSATION_TERM
        COMPENSATION_TYPE comp = compensation[f];
    #endif

    for (int r = 0; r < OUT_BLOCK_HEIGHT; r++)
    {
        #if OUTPUT_SIZE_Y % OUT_BLOCK_HEIGHT != 0
        const bool zero_r = or + r >= OUTPUT_SIZE_Y;
        if(!zero_r)
        #endif
        {
        for (int c = 0; c < OUT_BLOCK_WIDTH; c++)
        {
            #if OUTPUT_SIZE_X % OUT_BLOCK_WIDTH != 0
            const bool zero_c = oc + c >= OUTPUT_SIZE_X;
            if(!zero_c)
            #endif
            {
            #if OUTPUT_LAYOUT_B_FS_YX_FSV4 == 1
                uint out_idx = output_idx_offset + r * output_row_size_bytes + (c*PACK);
            #elif OUTPUT_LAYOUT_B_FS_YX_FSV16 == 1 || OUTPUT_LAYOUT_BS_FS_YX_BSV16_FSV16 == 1
                uint out_idx = OUTPUT_GET_INDEX(batch, f, or + r, oc + c);
            #else
                #error "Incorrect output layout"
            #endif
            ACCUMULATOR_TYPE dotProd = out[r * OUT_BLOCK_WIDTH + c];

            ACTIVATION_TYPE res = TO_ACTIVATION_TYPE(dotProd);

#if BIAS_TERM
    #if BIAS_PER_OUTPUT
                const uint bias_index = GET_DATA_INDEX(BIAS, batch, f, or + r, oc + c);
    #elif BIAS_PER_OFM
                const uint bias_index = f;
    #endif
                res += TO_ACTIVATION_TYPE(biases[bias_index]);
#endif

#ifdef COMPENSATION_TERM
                res += comp;
#endif

                OUTPUT_TYPE final_result;
#if HAS_FUSED_OPS
    #if FUSED_OPS_CAN_USE_PRELOAD
                FUSED_OPS_CALC;
    #else
                FUSED_OPS;
    #endif
                final_result = FUSED_OPS_RESULT;
#else
                final_result = TO_OUTPUT_TYPE(res);
#endif
#if FILTER_OFM_NUM % SIMD_SIZE != 0
                if (fmg % CEIL_DIV(FILTER_OFM_NUM, SIMD_SIZE) != CEIL_DIV(FILTER_OFM_NUM, SIMD_SIZE) - 1 || sglid < FILTER_OFM_NUM % SIMD_SIZE)
#endif
                    output[out_idx] = final_result;
            } // if(!zero_c)
        } // for (int c = 0; c < OUT_BLOCK_WIDTH; c++)
        } // if(!zero_r)
    } // for (int r = 0; r < OUT_BLOCK_HEIGHT; r++)
}

#if NON_BLOCK_LOAD != 1
#undef BLOCK_LOAD_WEIGHTS
#endif

#undef BLOCK_LOAD_INPUTS
#undef IN_BLOCK_WIDTH
#undef IN_BLOCK_HEIGHT
#undef PACK
#undef TYPE_N_
#undef TYPE_N
#undef AS_TYPE_N_
#undef AS_TYPE_N
#undef INPUT0_TYPE_4
#undef AS_INPUT0_TYPE_4
#undef NON_ZERO_INPUT0_PAD_BEFORE
#undef SHOULD_BALANCE_COMPENSATION
#undef SHOULD_USE_DATA_ZP
#undef SHOULD_USE_DATA_AND_WEIGHTS_ZP
#undef FILTER_TYPE_4
#undef AS_FILTER_TYPE_4
#undef NUM_FILTERS
