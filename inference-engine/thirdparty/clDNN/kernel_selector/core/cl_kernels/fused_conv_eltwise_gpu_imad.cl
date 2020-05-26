// Copyright (c) 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/common.cl"
#include "include/fetch.cl"
#include "include/imad.cl"
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

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)
#define AS_FILTER_TYPE_4(x) AS_TYPE_N(FILTER_TYPE, 4, x)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ALIGN(a, b) ((a % b == 0) ? a : a - a % b + b)

// int8 conv_input and weights data is packed to int32 "batches",
// int/uint pointers here instead of INPUT0_TYPE/FILTER_TYPE for convenience
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
KERNEL (fused_convolution_eltwise_gpu_imad)(
#if INPUT0_LAYOUT_B_FS_YX_FSV16
    const __global INPUT0_TYPE* conv_input,
#else
    const __global PACKED_TYPE   *conv_input,
#endif
    __global OUTPUT_TYPE         *restrict output,
    const __global int           *weights,
#if BIAS_TERM
    const __global BIAS_TYPE     *biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
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
    ACCUMULATOR_TYPE out[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT] = { 0 };  // this is the 32 bit signed accumulator that must be converted to 8 bits before final write.

    #define NUM_FILTERS (FILTER_SIZE_Y * FILTER_SIZE_X)
    int w[NUM_FILTERS];
    int in_addr;

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
#if INPUT0_LAYOUT_B_FS_YX_FSV16
        in_addr = INPUT0_GET_INDEX(batch, (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * PACK, input_y, input_x + sglid);
#else
    #ifdef BLOCK_LOAD_INPUTS
        in_addr = INPUT0_OFFSET + (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * INPUT0_FEATURE_PITCH + input_y * INPUT0_Y_PITCH + input_x;
    #else
        in_addr = INPUT0_OFFSET + (kd + g * CEIL_DIV(FILTER_IFM_NUM, PACK)) * INPUT0_FEATURE_PITCH + input_y * INPUT0_Y_PITCH + input_x + sglid;
    #endif
        in_addr += batch * input_size;  // adjust for batching
#endif
        for(uint reg = 0; reg < IN_BLOCK_HEIGHT; reg++) {
#if INPUT0_LAYOUT_B_FS_YX_FSV16
            in[reg] = *(__global PACKED_TYPE*)(conv_input + in_addr);
            in_addr += (INPUT0_SIZE_X + IWPAD) * 16;
#else
    #ifdef BLOCK_LOAD_INPUTS
            in[reg] = AS_PACKED_TYPE(intel_sub_group_block_read(&conv_input[in_addr]));
    #else
            in[reg] = AS_PACKED_TYPE(conv_input[in_addr]);// read SIMD_SIZE elements wide
    #endif
            // TODO This will cause errors for byxf_af32 format on input
            in_addr += (INPUT0_SIZE_X + IWPAD);  // move to next row down
#endif
        }

#ifdef BLOCK_LOAD_WEIGHTS
        *((int8*)&w[0]) = as_int8(intel_sub_group_block_read8((const __global uint*) &weights[weight_addr]));
        w[8]= as_int(intel_sub_group_block_read((const __global uint*) &weights[weight_addr + (SIMD_SIZE<<3)]));
        weight_addr += SIMD_SIZE*NUM_FILTERS;
#else
        for(int pf=0; pf < NUM_FILTERS; pf++) {
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
                __attribute__((opencl_unroll_hint))
                for (int br = 0; br < OUT_BLOCK_HEIGHT; br++) {
                    __attribute__((opencl_unroll_hint))
                    for (int bc = 0; bc < OUT_BLOCK_WIDTH; bc++) {
                        PACKED_TYPE input = sub_group_broadcast(in[br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y], bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X);

                        out[br * OUT_BLOCK_WIDTH + bc] = TO_ACCUMULATOR_TYPE(IMAD(out[br * OUT_BLOCK_WIDTH + bc], AS_INPUT0_TYPE_4(input), AS_FILTER_TYPE_4(w[wi])));
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
            #if OUTPUT_LAYOUT_BYXF_AF32 == 1
                uint out_idx = OUTPUT_GET_INDEX(batch, f, or + r, oc + c);
            #elif OUTPUT_LAYOUT_B_FS_YX_FSV4 == 1
                uint out_idx = output_idx_offset + r * output_row_size_bytes + (c*PACK);
            #elif OUTPUT_LAYOUT_B_FS_YX_FSV16 == 1 || OUTPUT_LAYOUT_BS_FS_YX_BSV16_FSV16 == 1
                uint out_idx = OUTPUT_GET_INDEX(batch, f, or + r, oc + c);
            #else
                #error "Incorrect output layout"
            #endif
            ACCUMULATOR_TYPE dotProd = out[r * OUT_BLOCK_WIDTH + c];

#if BIAS_TERM
    #if BIAS_PER_OUTPUT
                const uint bias_index = GET_DATA_INDEX(BIAS, batch, f, or + r, oc + c);
    #elif BIAS_PER_OFM
                const uint bias_index = f;
    #endif
                ACTIVATION_TYPE res = TO_ACTIVATION_TYPE(dotProd) + TO_ACTIVATION_TYPE(biases[bias_index]);
#else
                ACTIVATION_TYPE res = TO_ACTIVATION_TYPE(dotProd);
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
            }// if(!zero_c)
        } // for (int c = 0; c < OUT_BLOCK_WIDTH; c++)
        }// if(!zero_r)
    } // for (int r = 0; r < OUT_BLOCK_HEIGHT; r++)
}

#if NON_BLOCK_LOAD != 1
#undef BLOCK_LOAD_WEIGHTS
#endif

#undef BLOCK_LOAD_INPUTS
#undef IN_BLOCK_WIDTH
#undef IN_BLOCK_HEIGHT
#undef PACK
#undef AS_TYPE_N_
#undef AS_TYPE_N
#undef AS_INPUT0_TYPE_4
#undef AS_FILTER_TYPE_4
#undef NUM_FILTERS
#undef CEIL_DIV
#undef ALIGN
