// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_bfyx_iyxo_5x5)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weights
#if BIAS_TERM
    , const __global UNIT_TYPE* bias
#endif
)
{
    const uint idx = 4 * ((uint)get_global_id(0) * 16 + (uint)get_global_id(2));
    const uint idy = (uint)get_global_id(1);
    uint filter_idx = 0;
    uint output_idx = 0;
    uint input_idx = 0;
    UNIT_TYPE inp[8] = { 0 };

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
                            uint inp_idx = ifm * (INPUT0_FEATURE_PITCH)+(idy + yy) * (INPUT0_Y_PITCH)+idx;
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
                for (int ofm = 0; ofm < FILTER_OFM_MAX; ofm++) {
#if BIAS_TERM
                    out1[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
                    out2[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
                    out3[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
                    out4[ofm] += bias[(iter * FILTER_OFM_MAX) + ofm];
#endif
                    out1[ofm] = ACTIVATION(out1[ofm], ACTIVATION_PARAMS);
                    out2[ofm] = ACTIVATION(out2[ofm], ACTIVATION_PARAMS);
                    out3[ofm] = ACTIVATION(out3[ofm], ACTIVATION_PARAMS);
                    out4[ofm] = ACTIVATION(out4[ofm], ACTIVATION_PARAMS);
                    output_idx = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + ofm * OUTPUT_FEATURE_PITCH +
                        idy * OUTPUT_Y_PITCH + idx;
#if OUTPUT_OFFSET > 0
#if (OUTPUT_OFFSET % 2) > 0
                    output[output_idx + OUTPUT_OFFSET + 0] = out1[ofm];
                    output[output_idx + OUTPUT_OFFSET + 1] = out2[ofm];
                    output[output_idx + OUTPUT_OFFSET + 2] = out3[ofm];
                    output[output_idx + OUTPUT_OFFSET + 3] = out4[ofm];
#else
                    __global float* out_fl = output + output_idx + OUTPUT_OFFSET;
                    out_fl[0] = as_float((half2)(out1[ofm], out2[ofm]));
                    out_fl[1] = as_float((half2)(out3[ofm], out4[ofm]));
#endif
#else
                    vstore2((float2)(as_float((half2)(out1[ofm], out2[ofm])), as_float((half2)(out3[ofm], out4[ofm]))),
                        0, (__global float*)(output + output_idx));
#endif
                }
        }
}
