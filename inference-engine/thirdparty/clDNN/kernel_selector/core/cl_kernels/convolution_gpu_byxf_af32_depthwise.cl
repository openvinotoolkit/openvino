// Copyright (c) 2016-2020 Intel Corporation
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

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

#if STRIDE_SIZE_Y == DILATION_SIZE_Y
    #define BLOCK_Y_SIZE (FILTER_SIZE_Y + (SPLIT_Y - 1))
    #define LOAD_Y_WITH_STRIDES
#else
    #define BLOCK_Y_SIZE ((SPLIT_Y - 1) * STRIDE_SIZE_Y + (FILTER_SIZE_Y - 1) * (DILATION_SIZE_Y - 1) + FILTER_SIZE_Y)
#endif

#if STRIDE_SIZE_X == DILATION_SIZE_X
    #define FILTER_SIZE_X_PRELOAD FILTER_SIZE_X
    #define LOAD_X_WITH_STRIDES
#else
    #define FILTER_SIZE_X_PRELOAD FILTER_SIZE_X
    #define LOAD_X_WITH_STRIDES
    #define DONT_USE_X_SHIFTS
#endif

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL(convolution_gpu_byxf_af32_depthwise)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
{
    const uint x = get_global_id(1) * OUT_BLOCK_WIDTH;
    const uint y = get_global_id(2) * SPLIT_Y;
#if OUTPUT_BATCH_NUM == 1
    const uint of = get_global_id(0);
    const uint b = 0;
#else
    const uint of = (uint)get_global_id(0) % ALIGNED_OFM;
    const uint b = (uint)get_global_id(0) / ALIGNED_OFM;
#endif
    const uint g = of;

    if (of >= OUTPUT_FEATURE_NUM)
        return;

    int dotProd[SPLIT_Y] = {0};
    OUTPUT_TYPE out[SPLIT_Y];
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = g*FILTER_GROUPS_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + g*FILTER_IFM_NUM;

    // read all weights
    FILTER_TYPE w[FILTER_IFM_PITCH];
    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
    for (int j = 0; j < FILTER_SIZE_Y; j++) {
        __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
        for (int i = 0; i < FILTER_SIZE_X; i++) {
            w[j * FILTER_SIZE_X + i] = weights[filter_offset + j * FILTER_Y_PITCH + i * FILTER_X_PITCH];
        }
    }

    // initial input read
    INPUT0_TYPE in[FILTER_SIZE_X_PRELOAD * BLOCK_Y_SIZE];
    __attribute__((opencl_unroll_hint(BLOCK_Y_SIZE)))
    for (int i = 0; i < BLOCK_Y_SIZE; i++) {
        __attribute__((opencl_unroll_hint(FILTER_SIZE_X_PRELOAD)))
        for (int j = 0; j < FILTER_SIZE_X_PRELOAD; j++) {
#ifdef LOAD_Y_WITH_STRIDES
            int input_offset_y = input_y + i * DILATION_SIZE_Y;
#else
            int input_offset_y = input_y + i;
#endif
#ifdef LOAD_X_WITH_STRIDES
            int input_offset_x = input_x + j * DILATION_SIZE_X;
#else
            int input_offset_x = input_x + j;
#endif
            uint input_idx = input_offset + (uint)input_offset_x * INPUT0_X_PITCH + (uint)input_offset_y * INPUT0_Y_PITCH;
            in[i * FILTER_SIZE_X_PRELOAD + j] = input[input_idx];
        }
    }

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD;
#endif

    for (int l = 0; l < OUT_BLOCK_WIDTH; l++) {
        //calculate dotproduct
        __attribute__((opencl_unroll_hint(SPLIT_Y)))
        for (int i = 0; i < SPLIT_Y; i++) {
            __attribute__((opencl_unroll_hint(FILTER_IFM_PITCH)))
            for (int j = 0; j < FILTER_IFM_PITCH; j++) {
#if defined(LOAD_X_WITH_STRIDES) && defined(LOAD_Y_WITH_STRIDES)
                const uint start_pos_y = i * FILTER_SIZE_X_PRELOAD;
                dotProd[i] += (int)in[start_pos_y + j] * (int)w[j];
#elif defined(LOAD_X_WITH_STRIDES) && !defined(LOAD_Y_WITH_STRIDES)
                const uint start_pos_y = i * STRIDE_SIZE_Y * FILTER_SIZE_X_PRELOAD;
                const uint pos_y = start_pos_y + (j / FILTER_SIZE_X) * DILATION_SIZE_Y * FILTER_SIZE_X_PRELOAD;
                const uint pos_x = (j % FILTER_SIZE_X);
                dotProd[i] += (int)in[pos_y + pos_x] * (int)w[j];
#elif defined(LOAD_Y_WITH_STRIDES) && !defined(LOAD_X_WITH_STRIDES)
                const uint start_pos_y = i * FILTER_SIZE_X_PRELOAD;
                const uint pos_y = start_pos_y + (j / FILTER_SIZE_X) * FILTER_SIZE_X_PRELOAD;
                const uint pos_x = (j % FILTER_SIZE_X) * DILATION_SIZE_X;
                dotProd[i] += (int)in[pos_y + pos_x] * (int)w[j];
#else
                const uint start_pos_y = i * STRIDE_SIZE_Y * FILTER_SIZE_X_PRELOAD;
                const uint pos_y = start_pos_y + (j / FILTER_SIZE_X) * DILATION_SIZE_Y * FILTER_SIZE_X_PRELOAD;
                const uint pos_x = (j % FILTER_SIZE_X) * DILATION_SIZE_X;
                dotProd[i] += (int)in[pos_y + pos_x] * (int)w[j];
#endif  // defined(LOAD_X_WITH_STRIDES) && defined(LOAD_Y_WITH_STRIDES)
            }
        }

        __attribute__((opencl_unroll_hint(BLOCK_Y_SIZE)))
        for (int i = 0; i < BLOCK_Y_SIZE; i++) {
            // inputs shift
#ifndef DONT_USE_X_SHIFTS
#if (FILTER_SIZE_X_PRELOAD - STRIDE_SIZE_X) > 0
            __attribute__((opencl_unroll_hint(FILTER_SIZE_X_PRELOAD - STRIDE_SIZE_X)))
#endif
            for (int j = 0; j < FILTER_SIZE_X_PRELOAD - STRIDE_SIZE_X; j++) {
                in[i * FILTER_SIZE_X_PRELOAD + j] = in[i * FILTER_SIZE_X_PRELOAD + j + STRIDE_SIZE_X];
            }
#endif

            // read additional inputs
#ifdef LOAD_Y_WITH_STRIDES
            int input_offset_y = input_y + i * DILATION_SIZE_Y;
#else
            int input_offset_y = input_y + i;
#endif  // LOAD_Y_WITH_STRIDES

#if defined(DONT_USE_X_SHIFTS)
            __attribute__((opencl_unroll_hint(FILTER_SIZE_X_PRELOAD)))
            for (int j = 0; j < FILTER_SIZE_X_PRELOAD; j++) {
                int input_offset_x = input_x + ((l + 1) * STRIDE_SIZE_X) + j * DILATION_SIZE_X;
                uint input_idx = input_offset + (uint)input_offset_x * INPUT0_X_PITCH + (uint)input_offset_y * INPUT0_Y_PITCH;
                in[i * FILTER_SIZE_X_PRELOAD + j] = input[input_idx];
            }

#else
            {
                int input_offset_x = input_x + ((l + 1) * STRIDE_SIZE_X) + (FILTER_SIZE_X - 1) * DILATION_SIZE_X;
                uint input_idx = input_offset + (uint)input_offset_x * INPUT0_X_PITCH + (uint)input_offset_y * INPUT0_Y_PITCH;
                in[i * FILTER_SIZE_X_PRELOAD + FILTER_SIZE_X_PRELOAD - 1] = input[input_idx];
            }
#endif  // defined(DONT_USE_X_SHIFTS)
        }

        __attribute__((opencl_unroll_hint(SPLIT_Y)))
        for (int m = 0; m < SPLIT_Y; m++) {
#if BIAS_TERM
        #if BIAS_PER_OUTPUT
        #if OUTPUT_LAYOUT_BYXF_AF32 == 1
            const uint bias_index = GET_DATA_INDEX(BIAS, b, of, y + m, x + l);
        #elif OUTPUT_LAYOUT_B_FS_YX_FSV4 == 1
            const uint bias_index = GET_DATA_B_FS_YX_FSV4_INDEX(BIAS, b, of, y + m, x + l);
        #else
            #error "Incorrect output layout"
        #endif
#elif BIAS_PER_OFM
            const uint bias_index = of;
#endif
            // TODO: Maybe half should be supported as well.
            float res = (float)dotProd[m] + biases[bias_index];
#else
            float res = (float)dotProd[m];
#endif
            dotProd[m] = 0;

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_CALC;
#else
            FUSED_OPS;
#endif
            out[m] = FUSED_OPS_RESULT;
#else
            out[m] = TO_OUTPUT_TYPE(res);
#endif
        }

        __attribute__((opencl_unroll_hint(SPLIT_Y)))
        for (int m = 0; m < SPLIT_Y; m++) {
#ifdef SPLIT_LEFTOVERS
            if (y + m >= OUTPUT_SIZE_Y)
                continue;
#endif
            const uint dst_index = OUTPUT_GET_INDEX(b, of, y + m, x + l);
            output[dst_index] = ACTIVATION(out[m], ACTIVATION_PARAMS);
        }
    }  // OUT_BLOCK_WIDTH
}
