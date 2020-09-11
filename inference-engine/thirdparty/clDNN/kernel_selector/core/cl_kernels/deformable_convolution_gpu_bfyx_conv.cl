// Copyright (c) 2019 Intel Corporation
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

#include "include/include_all.cl"
#include "include/unit_type.cl"

#define FEATURE_SLICE_SIZE 16

#define GET_WEI(filter, id) AS_TYPE(UNIT_TYPE, intel_sub_group_shuffle(AS_TYPE(UNIT_BLOCK_RW_TYPE, filter), id))

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(deformable_convolution_gpu_bfyx_conv)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx)
{
    const uint lid = get_sub_group_local_id();
    const uint x = ((uint)get_global_id(0) * X_BLOCK_SIZE + lid) % OUTPUT_SIZE_X;
    const uint y = ((uint)get_global_id(0) * X_BLOCK_SIZE + lid) / OUTPUT_SIZE_X;
    const uint f_block = get_group_id(1);
    const uint b = get_global_id(2);

    UNIT_TYPE dotProd[16] = { UNIT_VAL_ZERO };

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
    const uint filter_offset = f_block * filter_os_pitch;

    const uint input_offset = INPUT0_OFFSET + b*INPUT0_BATCH_PITCH + x*INPUT0_X_PITCH + y*INPUT0_Y_PITCH;
    const uint input_kh_pitch = FILTER_SIZE_X*INPUT_CHANNELS*INPUT0_FEATURE_PITCH;
    const uint input_kw_pitch = INPUT_CHANNELS*INPUT0_FEATURE_PITCH;

    for (uint kh = 0; kh < FILTER_SIZE_Y ; ++kh)
    {
        for (uint kw = 0; kw < FILTER_SIZE_X ; ++kw)
        {
            for (uint icb = 0; icb < (INPUT_CHANNELS + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE; ++icb)
            {
                UNIT_TYPE8 wei0 = UNIT_BLOCK_READ8(weights, filter_offset +
                                                            icb * filter_is_pitch +
                                                            kh * filter_y_pitch +
                                                            kw * filter_x_pitch);
                UNIT_TYPE8 wei1 = UNIT_BLOCK_READ8(weights, filter_offset +
                                                            icb * filter_is_pitch +
                                                            kh * filter_y_pitch +
                                                            kw * filter_x_pitch +
                                                            8 * filter_isv_pitch);

                UNIT_TYPE src[16];
                for (int ic = 0; ic < 16; ic++) {
                    if (icb*FEATURE_SLICE_SIZE + ic < INPUT_CHANNELS)
                        src[ic] = input[input_offset +
                                        kh*input_kh_pitch +
                                        kw*input_kw_pitch +
                                        (icb*FEATURE_SLICE_SIZE + ic)*INPUT0_FEATURE_PITCH];
                    else
                        src[ic] = 0.0f;
                }

                for (int oc = 0; oc < 16; oc++) {
                    dotProd[oc] += src[0] * GET_WEI(wei0.s0, oc);
                    dotProd[oc] += src[1] * GET_WEI(wei0.s1, oc);
                    dotProd[oc] += src[2] * GET_WEI(wei0.s2, oc);
                    dotProd[oc] += src[3] * GET_WEI(wei0.s3, oc);
                    dotProd[oc] += src[4] * GET_WEI(wei0.s4, oc);
                    dotProd[oc] += src[5] * GET_WEI(wei0.s5, oc);
                    dotProd[oc] += src[6] * GET_WEI(wei0.s6, oc);
                    dotProd[oc] += src[7] * GET_WEI(wei0.s7, oc);
                    dotProd[oc] += src[8] * GET_WEI(wei1.s0, oc);
                    dotProd[oc] += src[9] * GET_WEI(wei1.s1, oc);
                    dotProd[oc] += src[10] * GET_WEI(wei1.s2, oc);
                    dotProd[oc] += src[11] * GET_WEI(wei1.s3, oc);
                    dotProd[oc] += src[12] * GET_WEI(wei1.s4, oc);
                    dotProd[oc] += src[13] * GET_WEI(wei1.s5, oc);
                    dotProd[oc] += src[14] * GET_WEI(wei1.s6, oc);
                    dotProd[oc] += src[15] * GET_WEI(wei1.s7, oc);
                }
            }
        }
    }

    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f_block*FEATURE_SLICE_SIZE, y, x);
    for (int oc = 0; oc < 16; oc++)
    {
#if BIAS_TERM
        const uint bias_index = f_block*FEATURE_SLICE_SIZE + oc;
        dotProd[oc] += (UNIT_TYPE)biases[bias_index];
#endif
        if ((uint)get_global_id(0) * X_BLOCK_SIZE + lid < OUTPUT_SIZE_X*OUTPUT_SIZE_Y && f_block*FEATURE_SLICE_SIZE + oc < OUTPUT_FEATURE_NUM)
            output[dst_index + oc*OUTPUT_FEATURE_PITCH] = ACTIVATION(dotProd[oc], ACTIVATION_PARAMS);
    }

}
