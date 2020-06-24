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
#include "include/imad.cl"

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)
#define AS_FILTER_TYPE_4(x) AS_TYPE_N(FILTER_TYPE, 4, x)

#define ACTIVATION_TYPE half
#define TO_ACTIVATION_TYPE(x) convert_half(x)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(fused_conv_eltwise_kernel_b_fs_yx_fsv4)(
    const __global float* input,
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
    write_only image2d_t output,
#else
    __global OUTPUT_TYPE* output,
#endif
    const __global UNIT_TYPE* weights,
#if BIAS_TERM
    const __global BIAS_TYPE* bias,
#endif
    uint split_idx,
    const __global INPUT1_TYPE* eltw_input
#if HAS_FUSED_OPS_DECLS
    ,FUSED_OPS_DECLS
#endif
    )
{
    const uint idx = ((uint)get_global_id(0) * 16 + (uint)get_global_id(2)) * 2;
    const uint idy = (uint)get_global_id(1);
    uint filter_idx = 0;
    uint input_idx = 0;
    int batch = 0;
    uint output_idx_eltwise = 0;

#if FILTER_OFM_NUM > 16
#define FILTER_OFM_MAX 16
#else
#define FILTER_OFM_MAX FILTER_OFM_NUM
#endif
    __attribute__((opencl_unroll_hint(FILTER_OFM_NUM / FILTER_OFM_MAX)))
        for (int iter = 0; iter < FILTER_OFM_NUM / FILTER_OFM_MAX + (FILTER_OFM_NUM % FILTER_OFM_MAX != 0); iter++) {
            int out1[FILTER_OFM_MAX] = { 0 };
            int out2[FILTER_OFM_MAX] = { 0 };

            filter_idx = FILTER_OFM_MAX * iter * 4;

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
                                            __global uint* www = weights + (filter_idx + ofm * 4);
                                            char4 w = AS_FILTER_TYPE_4(www[0]);
                                            out1[ofm] = IMAD(out1[ofm], AS_INPUT0_TYPE_4((char4)(tmp[0], tmp[1], tmp[2], tmp[3])), AS_FILTER_TYPE_4(w));
                                            out2[ofm] = IMAD(out2[ofm], AS_INPUT0_TYPE_4((char4)(tmp[4], tmp[5], tmp[6], tmp[7])), AS_FILTER_TYPE_4(w));
                                        }
                                    filter_idx += 64;
                                }
                        }
                }

            
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
            __attribute__((opencl_unroll_hint(FILTER_OFM_MAX/3)))
                for (int ofm = 0; ofm < FILTER_OFM_MAX; ofm+=3) {
#else
            __attribute__((opencl_unroll_hint(FILTER_OFM_MAX)))
                for (int ofm = 0; ofm < FILTER_OFM_MAX; ofm++) {
#endif
#if BIAS_TERM
                    ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(out1[ofm + 0]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 0]);
                    ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(out2[ofm + 0]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 0]);
                    res0 = ACTIVATION(res0, ACTIVATION_PARAMS);
                    res1 = ACTIVATION(res1, ACTIVATION_PARAMS);
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                    ACTIVATION_TYPE res2 = TO_ACTIVATION_TYPE(out1[ofm + 1]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 1]);
                    ACTIVATION_TYPE res3 = TO_ACTIVATION_TYPE(out2[ofm + 1]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 1]);
                    ACTIVATION_TYPE res4 = TO_ACTIVATION_TYPE(out1[ofm + 2]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 2]);
                    ACTIVATION_TYPE res5 = TO_ACTIVATION_TYPE(out2[ofm + 2]) + TO_ACTIVATION_TYPE(bias[(iter * FILTER_OFM_MAX) + ofm + 2]);
                    res2 = ACTIVATION(res2, ACTIVATION_PARAMS);
                    res3 = ACTIVATION(res3, ACTIVATION_PARAMS);
                    res4 = ACTIVATION(res4, ACTIVATION_PARAMS);
                    res5 = ACTIVATION(res5, ACTIVATION_PARAMS);
#endif
#endif
                    ACTIVATION_TYPE pack0 = 0;
                    ACTIVATION_TYPE pack1 = 0;
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                    ACTIVATION_TYPE pack2 = 0;
                    ACTIVATION_TYPE pack3 = 0;
                    ACTIVATION_TYPE pack4 = 0;
                    ACTIVATION_TYPE pack5 = 0;
#endif
                    {
                        FUSED_OP0_LOAD_0;
                        pack0 = convert_float(scale0_data1) * res0;
                        pack1 = convert_float(scale0_data1) * res1;
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                        pack2 = convert_float(scale0_data1) * res2;
                        pack3 = convert_float(scale0_data1) * res3;
                        pack4 = convert_float(scale0_data1) * res4;
                        pack5 = convert_float(scale0_data1) * res5;
#endif
                    };
                    uint ofm_alignment = 4;
                    int idx_for_image = 0;
                    int idy_for_image = 0;
                    int idx_adjust = 1;

                    if (ofm / OUTPUT_FEATURE_NUM == 0) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            2 * idy * OUTPUT_Y_PITCH + 2 * idx;
                        idx_for_image = 2 * idx;
                        idy_for_image = 2 * idy;
                        idx_adjust = 0;
                    }
                    else if (ofm / OUTPUT_FEATURE_NUM == 1) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            2 * idy * OUTPUT_Y_PITCH + 2 * idx + idx_adjust;
                        idx_for_image = 2 * idx + 1;
                        idy_for_image = 2 * idy;
                    }
                    else if (ofm / OUTPUT_FEATURE_NUM == 2) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            (2 * idy + 1) * OUTPUT_Y_PITCH + 2 * idx;
                        idx_for_image = 2 * idx;
                        idy_for_image = 2 * idy + 1;
                        idx_adjust = 0;
                    }
                    else if (ofm / OUTPUT_FEATURE_NUM == 3) {
                        output_idx_eltwise = (iter * FILTER_OFM_MAX * OUTPUT_FEATURE_PITCH) + (ofm % OUTPUT_FEATURE_NUM) * OUTPUT_FEATURE_PITCH +
                            (2 * idy + 1) * OUTPUT_Y_PITCH + 2 * idx + idx_adjust;
                        idx_for_image = 2 * idx + 1;
                        idy_for_image = 2 * idy + 1;
                    }
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
                    half4 tmp_elt0 = as_half4(vload2(0, (__global uint*)(eltw_input + output_idx_eltwise + OUTPUT_OFFSET - idx_adjust + OUTPUT_FEATURE_PITCH * 0)));
                    half4 tmp_elt1 = as_half4(vload2(0, (__global uint*)(eltw_input + output_idx_eltwise + OUTPUT_OFFSET - idx_adjust + OUTPUT_FEATURE_PITCH * 1)));
                    half4 tmp_elt2 = as_half4(vload2(0, (__global uint*)(eltw_input + output_idx_eltwise + OUTPUT_OFFSET - idx_adjust + OUTPUT_FEATURE_PITCH * 2)));
                    half4 output_half1 = {
                        pack0 + tmp_elt0[0 + idx_adjust],
                        pack2 + tmp_elt1[0 + idx_adjust],
                        pack4 + tmp_elt2[0 + idx_adjust],
                        0 };
                    IMAGE_WRITE(output, (int2)(idx_for_image, idy_for_image), output_half1);
                    half4 output_half2 = {
                        pack1 + tmp_elt0[2 + idx_adjust],
                        pack3 + tmp_elt1[2 + idx_adjust],
                        pack5 + tmp_elt2[2 + idx_adjust],
                        0 };
                    IMAGE_WRITE(output, (int2)(idx_for_image +2, idy_for_image), output_half2);
#else
                    half4 tmp_elt0 = as_half4(vload2(0, (__global uint*)(eltw_input + output_idx_eltwise + OUTPUT_OFFSET - idx_adjust)));
                    output[output_idx_eltwise + OUTPUT_OFFSET + 0] = pack0 + tmp_elt0[0 + idx_adjust];
                    output[output_idx_eltwise + OUTPUT_OFFSET + 2] = pack1 + tmp_elt0[2 + idx_adjust];
#endif
                }
        }
}
