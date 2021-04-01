// Copyright (c) 2020 Intel Corporation
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

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(fused_conv_eltwise_gpu_bfyx_iyxo)(
    const __global UNIT_TYPE* input,
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
    write_only image2d_t output,
#else
    __global UNIT_TYPE* output,
#endif
    const __global UNIT_TYPE* weights,
#if BIAS_TERM
    const __global UNIT_TYPE* bias,
#endif
    uint split_idx,
    const __global UNIT_TYPE* eltw_input)
{
    const uint idx = 4 * ((uint)get_global_id(0) * 16 + (uint)get_global_id(2));
    const uint idy = (uint)get_global_id(1);
    uint filter_idx = 0;
    uint output_idx = 0;
    uint output_idx_eltwise = 0;
    uint input_idx = 0;
    UNIT_TYPE inp[8] = { 0 };
    const uint input0_pitch_Y = INPUT0_SIZE_X + 2 * (INPUT0_PAD_BEFORE_SIZE_X);
    const uint input0_pitch_feature = input0_pitch_Y * (INPUT0_SIZE_Y + 2 * (INPUT0_PAD_BEFORE_SIZE_Y));

#if FILTER_OFM_NUM > 16
#define FILTER_OFM_MAX 16
#else
#define FILTER_OFM_MAX FILTER_OFM_NUM
#endif
    __attribute__((opencl_unroll_hint(1)))
        for (int iter = 0; iter < FILTER_OFM_NUM / FILTER_OFM_MAX + (FILTER_OFM_NUM % FILTER_OFM_MAX != 0); iter++) {
            UNIT_TYPE out1[FILTER_OFM_MAX] = { 0 };
            UNIT_TYPE out2[FILTER_OFM_MAX] = { 0 };
            UNIT_TYPE out3[FILTER_OFM_MAX] = { 0 };
            UNIT_TYPE out4[FILTER_OFM_MAX] = { 0 };

            filter_idx = FILTER_OFM_MAX * iter;

            __attribute__((opencl_unroll_hint(FILTER_IFM_NUM)))
                for (int ifm = 0; ifm < FILTER_IFM_NUM; ifm++) {
                    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
                        for (int yy = 0; yy < FILTER_SIZE_Y; yy++) {
                            uint inp_idx = ifm * input0_pitch_feature + (idy + yy) * input0_pitch_Y + idx;
                            half8 tmp = as_half8(vload4(0, (__global uint*)(input + inp_idx)));

                            inp[0] = tmp.s0;
                            inp[1] = tmp.s1;
                            inp[2] = tmp.s2;
                            inp[3] = tmp.s3;
                            inp[4] = tmp.s4;
                            inp[5] = tmp.s5;
                            inp[6] = tmp.s6;
                            inp[7] = tmp.s7;

                            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
                                for (int xx = 0; xx < FILTER_SIZE_X; xx++) {
#if FILTER_OFM_NUM == 4
                                    half4 w = as_half4(vload2(0, (__global uint*)(weights + filter_idx)));
#elif FILTER_OFM_NUM == 8
                                    half8 w = as_half8(vload4(0, (__global uint*)(weights + filter_idx)));
#else
                                    half16 w = as_half16(vload8(0, (__global uint*)(weights + filter_idx)));
#endif
                                    __attribute__((opencl_unroll_hint(FILTER_OFM_MAX)))
                                        for (int ofm = 0; ofm < FILTER_OFM_MAX; ofm++) {
                                            out1[ofm] = mad(inp[0 + xx], w[ofm], out1[ofm]);
                                            out2[ofm] = mad(inp[1 + xx], w[ofm], out2[ofm]);
                                            out3[ofm] = mad(inp[2 + xx], w[ofm], out3[ofm]);
                                            out4[ofm] = mad(inp[3 + xx], w[ofm], out4[ofm]);
                                        }
                                    filter_idx += FILTER_OFM_NUM;
                                }
                        }
                }

            __attribute__((opencl_unroll_hint(FILTER_OFM_MAX)))
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                for (int ofm = 0; ofm < FILTER_OFM_MAX; ofm+=3) {
#else
                for (int ofm = 0; ofm < FILTER_OFM_MAX; ofm++) {
#endif
#if BIAS_TERM
                    out1[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
                    out2[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
                    out3[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
                    out4[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                    out1[ofm + 1] += bias[(iter * FILTER_OFM_MAX) + ofm + 1];
                    out2[ofm + 1] += bias[(iter * FILTER_OFM_MAX) + ofm + 1];
                    out3[ofm + 1] += bias[(iter * FILTER_OFM_MAX) + ofm + 1];
                    out4[ofm + 1] += bias[(iter * FILTER_OFM_MAX) + ofm + 1];

                    out1[ofm + 2] += bias[(iter * FILTER_OFM_MAX) + ofm + 2];
                    out2[ofm + 2] += bias[(iter * FILTER_OFM_MAX) + ofm + 2];
                    out3[ofm + 2] += bias[(iter * FILTER_OFM_MAX) + ofm + 2];
                    out4[ofm + 2] += bias[(iter * FILTER_OFM_MAX) + ofm + 2];
#endif
#endif
                    out1[ofm] = ACTIVATION(out1[ofm], ACTIVATION_PARAMS);
                    out2[ofm] = ACTIVATION(out2[ofm], ACTIVATION_PARAMS);
                    out3[ofm] = ACTIVATION(out3[ofm], ACTIVATION_PARAMS);
                    out4[ofm] = ACTIVATION(out4[ofm], ACTIVATION_PARAMS);
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                    out1[ofm + 1] = ACTIVATION(out1[ofm + 1], ACTIVATION_PARAMS);
                    out2[ofm + 1] = ACTIVATION(out2[ofm + 1], ACTIVATION_PARAMS);
                    out3[ofm + 1] = ACTIVATION(out3[ofm + 1], ACTIVATION_PARAMS);
                    out4[ofm + 1] = ACTIVATION(out4[ofm + 1], ACTIVATION_PARAMS);

                    out1[ofm + 2] = ACTIVATION(out1[ofm + 2], ACTIVATION_PARAMS);
                    out2[ofm + 2] = ACTIVATION(out2[ofm + 2], ACTIVATION_PARAMS);
                    out3[ofm + 2] = ACTIVATION(out3[ofm + 2], ACTIVATION_PARAMS);
                    out4[ofm + 2] = ACTIVATION(out4[ofm + 2], ACTIVATION_PARAMS);
#endif
                    uint ofm_alignment = 4;
                    int idx_for_image = 0;
                    int idy_for_image = 0;

                    if (ofm / OUTPUT_FEATURE_NUM == 0) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            2 * idy * OUTPUT_Y_PITCH + 2 * idx;
                        output_idx = (ofm % OUTPUT_FEATURE_NUM) + 2 * idy * OUTPUT_SIZE_X * ofm_alignment + 2 * idx * ofm_alignment;
                        idx_for_image = 2 * idx;
                        idy_for_image = 2 * idy;
                    }
                    else if (ofm / OUTPUT_FEATURE_NUM == 1) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            2 * idy * OUTPUT_Y_PITCH + 2 * idx + 1;
                        output_idx = (ofm % OUTPUT_FEATURE_NUM) + 2 * idy * OUTPUT_SIZE_X * ofm_alignment + (2 * idx + 1) * ofm_alignment;
                        idx_for_image = 2 * idx + 1;
                        idy_for_image = 2 * idy;
                    }
                    else if (ofm / OUTPUT_FEATURE_NUM == 2) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            (2 * idy + 1) * OUTPUT_Y_PITCH + 2 * idx;
                        output_idx = (ofm % OUTPUT_FEATURE_NUM) + (2 * idy + 1) * OUTPUT_SIZE_X * ofm_alignment + 2 * idx * ofm_alignment;
                        idx_for_image = 2 * idx;
                        idy_for_image = 2 * idy + 1;
                    }
                    else if (ofm / OUTPUT_FEATURE_NUM == 3) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            (2 * idy + 1) * OUTPUT_Y_PITCH + 2 * idx + 1;
                        output_idx = (ofm % OUTPUT_FEATURE_NUM) + (2 * idy + 1) * OUTPUT_SIZE_X * ofm_alignment + (2 * idx + 1) * ofm_alignment;
                        idx_for_image = 2 * idx + 1;
                        idy_for_image = 2 * idy + 1;
                    }
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                    half4 output_half1 = {
                        out1[ofm + 0] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 0 + OUTPUT_FEATURE_PITCH * 0],
                        out1[ofm + 1] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 0 + OUTPUT_FEATURE_PITCH * 1],
                        out1[ofm + 2] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 0 + OUTPUT_FEATURE_PITCH * 2],
                        0 };
                    IMAGE_WRITE(output, (int2)(idx_for_image, idy_for_image), output_half1);
                    half4 output_half2 = {
                        out2[ofm + 0] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 2 + OUTPUT_FEATURE_PITCH * 0],
                        out2[ofm + 1] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 2 + OUTPUT_FEATURE_PITCH * 1],
                        out2[ofm + 2] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 2 + OUTPUT_FEATURE_PITCH * 2],
                        0 };
                    IMAGE_WRITE(output, (int2)(idx_for_image +2, idy_for_image), output_half2);
                    half4 output_half3 = {
                        out3[ofm + 0] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 4 + OUTPUT_FEATURE_PITCH * 0],
                        out3[ofm + 1] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 4 + OUTPUT_FEATURE_PITCH * 1],
                        out3[ofm + 2] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 4 + OUTPUT_FEATURE_PITCH * 2],
                        0 };
                    IMAGE_WRITE(output, (int2)(idx_for_image+4, idy_for_image), output_half3);
                    half4 output_half4 = {
                        out4[ofm + 0] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 6 + OUTPUT_FEATURE_PITCH * 0],
                        out4[ofm + 1] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 6 + OUTPUT_FEATURE_PITCH * 1],
                        out4[ofm + 2] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 6 + OUTPUT_FEATURE_PITCH * 2],
                        0 };
                    IMAGE_WRITE(output, (int2)(idx_for_image+6, idy_for_image), output_half4);
#else
                    output[output_idx_eltwise + OUTPUT_OFFSET + 0] = out1[ofm] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 0];
                    output[output_idx_eltwise + OUTPUT_OFFSET + 2] = out2[ofm] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 2];
                    output[output_idx_eltwise + OUTPUT_OFFSET + 4] = out3[ofm] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 4];
                    output[output_idx_eltwise + OUTPUT_OFFSET + 6] = out4[ofm] + eltw_input[output_idx_eltwise + OUTPUT_OFFSET + 6];
#endif
                }
        }
}
