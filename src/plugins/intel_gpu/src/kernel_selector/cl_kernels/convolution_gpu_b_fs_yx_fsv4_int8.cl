// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/imad.cl"

#define INPUT0_PACKED_TYPE uint

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_b_fs_yx_fsv4_int8)(
    const __global INPUT0_PACKED_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* bias
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)
#define AS_FILTER_TYPE_4(x) AS_TYPE_N(FILTER_TYPE, 4, x)

    const uint items_per_EU = 2;
    const uint idx = items_per_EU * ((uint)get_global_id(0) * SUB_GROUP_SIZE + (uint)get_global_id(2));
    const uint idy = (uint)get_global_id(1);
    uint filter_idx = 0;
    uint output_idx = 0;
    uint input_idx = 0;
    int batch = 0;
    const uint packed_values = 4;
#if FILTER_OFM_NUM > 8
#define FILTER_OFM_MAX 8
#else
#define FILTER_OFM_MAX FILTER_OFM_NUM
#endif
    __attribute__((opencl_unroll_hint(FILTER_OFM_NUM / FILTER_OFM_MAX)))
        for (int iter = 0; iter < FILTER_OFM_NUM / FILTER_OFM_MAX + (FILTER_OFM_NUM % FILTER_OFM_MAX != 0); iter++) {
            int out1[FILTER_OFM_MAX] = { 0 };
            int out2[FILTER_OFM_MAX] = { 0 };
            filter_idx = FILTER_OFM_MAX * iter * packed_values;

            __attribute__((opencl_unroll_hint(1)))
                for (int ifm = 0; ifm < (FILTER_IFM_NUM + 3) / 4; ifm++) {
                    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
                        for (int yy = 0; yy < FILTER_SIZE_Y; yy++) {
                            uint inp_idx = ifm * (INPUT0_FEATURE_PITCH)+(idy + yy) * (INPUT0_Y_PITCH)+idx;

                            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
                                for (int xx = 0; xx < FILTER_SIZE_X; xx++) {
                                    char8 tmp = as_char8(vload2(0, (__global uint*)(input + inp_idx + xx)));

                                    __attribute__((opencl_unroll_hint(FILTER_OFM_MAX)))
                                        for (int ofm = 0; ofm < FILTER_OFM_MAX; ofm++) {
                                            __global uint* www = weights + (filter_idx + ofm * packed_values);
                                            char4 w = AS_FILTER_TYPE_4(www[0]);
                                            out1[ofm] = IMAD(out1[ofm], AS_INPUT0_TYPE_4((char4)(tmp[0], tmp[1], tmp[2], tmp[3])), AS_FILTER_TYPE_4(w));
                                            out2[ofm] = IMAD(out2[ofm], AS_INPUT0_TYPE_4((char4)(tmp[4], tmp[5], tmp[6], tmp[7])), AS_FILTER_TYPE_4(w));
                                        }
                                    filter_idx += (packed_values * SUB_GROUP_SIZE);
                                }
                        }
                }

            __attribute__((opencl_unroll_hint(FILTER_OFM_MAX / 4)))
                for (int ofm = 0; ofm < FILTER_OFM_MAX && (ofm + iter * FILTER_OFM_MAX) < FILTER_OFM_NUM; ofm += packed_values) {

#if BIAS_TERM
                    ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(out1[ofm + 0]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 0]);
                    ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(out1[ofm + 1]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 1]);
                    ACTIVATION_TYPE res2 = TO_ACTIVATION_TYPE(out1[ofm + 2]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 2]);
                    ACTIVATION_TYPE res3 = TO_ACTIVATION_TYPE(out1[ofm + 3]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 3]);
                    ACTIVATION_TYPE res4 = TO_ACTIVATION_TYPE(out2[ofm + 0]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 0]);
                    ACTIVATION_TYPE res5 = TO_ACTIVATION_TYPE(out2[ofm + 1]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 1]);
                    ACTIVATION_TYPE res6 = TO_ACTIVATION_TYPE(out2[ofm + 2]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 2]);
                    ACTIVATION_TYPE res7 = TO_ACTIVATION_TYPE(out2[ofm + 3]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 3]);
#else
                    ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(out1[ofm + 0]);
                    ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(out1[ofm + 1]);
                    ACTIVATION_TYPE res2 = TO_ACTIVATION_TYPE(out1[ofm + 2]);
                    ACTIVATION_TYPE res3 = TO_ACTIVATION_TYPE(out1[ofm + 3]);
                    ACTIVATION_TYPE res4 = TO_ACTIVATION_TYPE(out2[ofm + 0]);
                    ACTIVATION_TYPE res5 = TO_ACTIVATION_TYPE(out2[ofm + 1]);
                    ACTIVATION_TYPE res6 = TO_ACTIVATION_TYPE(out2[ofm + 2]);
                    ACTIVATION_TYPE res7 = TO_ACTIVATION_TYPE(out2[ofm + 3]);
#endif

                    if (OUTPUT_PAD_BEFORE_FEATURE_NUM > 0) {
                        uint output_feature_specific_offset = OUTPUT_Y_PITCH * OUTPUT_PAD_BEFORE_SIZE_Y +
                            (OUTPUT_PAD_BEFORE_SIZE_X * OUTPUT_X_PITCH);
                        output_idx = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + ofm * OUTPUT_FEATURE_PITCH +
                            idy * OUTPUT_Y_PITCH * packed_values + idx * packed_values + OUTPUT_OFFSET + output_feature_specific_offset * 3;
                    }
                    else {
                        output_idx = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + ofm * OUTPUT_FEATURE_PITCH +
                            idy * OUTPUT_Y_PITCH * packed_values + idx * packed_values + packed_values * OUTPUT_OFFSET;
                    }

                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) pack1;
                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) pack2;
#if HAS_FUSED_OPS
                    { FUSED_OPS_0; pack1[0] = FUSED_OPS_RESULT_0; };
                    { FUSED_OPS_1; pack1[1] = FUSED_OPS_RESULT_1; };
                    { FUSED_OPS_2; pack1[2] = FUSED_OPS_RESULT_2; };
                    { FUSED_OPS_3; pack1[3] = FUSED_OPS_RESULT_3; };

                    { FUSED_OPS_4; pack2[0] = FUSED_OPS_RESULT_4; };
                    { FUSED_OPS_5; pack2[1] = FUSED_OPS_RESULT_5; };
                    { FUSED_OPS_6; pack2[2] = FUSED_OPS_RESULT_6; };
                    { FUSED_OPS_7; pack2[3] = FUSED_OPS_RESULT_7; };
#else
                    pack1[0] = TO_OUTPUT_TYPE(res0);
                    pack1[1] = TO_OUTPUT_TYPE(res1);
                    pack1[2] = TO_OUTPUT_TYPE(res2);
                    pack1[3] = TO_OUTPUT_TYPE(res3);
                    pack2[0] = TO_OUTPUT_TYPE(res4);
                    pack2[1] = TO_OUTPUT_TYPE(res5);
                    pack2[2] = TO_OUTPUT_TYPE(res6);
                    pack2[3] = TO_OUTPUT_TYPE(res7);
#endif

#if OUTPUT_TYPE_SIZE == 1
                    vstore2((float2)(
                        as_float(pack1),
                        as_float(pack2)
                        ), 0, (__global float*)(output + output_idx));
#else
#if OUTPUT_OFFSET % 4
                    output[output_idx + 0] = TO_OUTPUT_TYPE(pack1[0]);
                    output[output_idx + 1] = TO_OUTPUT_TYPE(pack1[1]);
                    output[output_idx + 2] = TO_OUTPUT_TYPE(pack1[2]);
                    output[output_idx + 3] = TO_OUTPUT_TYPE(pack1[3]);
                    output[output_idx + 4] = TO_OUTPUT_TYPE(pack2[0]);
                    output[output_idx + 5] = TO_OUTPUT_TYPE(pack2[1]);
                    output[output_idx + 6] = TO_OUTPUT_TYPE(pack2[2]);
                    output[output_idx + 7] = TO_OUTPUT_TYPE(pack2[3]);
#else
                    vstore4((float4)(TO_OUTPUT_TYPE(pack1[0]), TO_OUTPUT_TYPE(pack1[1]), TO_OUTPUT_TYPE(pack1[2]),
                        TO_OUTPUT_TYPE(pack1[3])), 0, (__global float*)(output + output_idx));
                    vstore4((float4)(TO_OUTPUT_TYPE(pack2[0]), TO_OUTPUT_TYPE(pack2[1]), TO_OUTPUT_TYPE(pack2[2]),
                        TO_OUTPUT_TYPE(pack2[3])), 0, (__global float*)(output + output_idx + 4));
#endif
#endif
                }
        }
}
