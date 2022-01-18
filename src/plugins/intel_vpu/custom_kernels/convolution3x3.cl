// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void Convolution3x3(
    const __global half *in_param,
    const __global half *out,
    const __global half *w,
    int IW,
    int IH,
    int IC,
    int OW,
    int OH,
    int OC,
    int KX,
    int KY,
    int stride_x,
    int stride_y,
    int pad_x,
    int pad_y,
    int dilation_x,
    int dilation_y)
{
    __local half in_local[8 * 1024];
    __local half out_local[8 * 1024];
    __local half w_local[8 * 1024];

    const int sizePlane = IW * IH;
    event_t e1          = async_work_group_copy_2D2D(
        in_local, // dst
        in_param + get_group_id(0) * stride_y * IW, // src
        3 * IW, // num_elements_per_line,
        IC, // num_lines,
        IW * IH - 3 * IW, // src_line_stride,
        0, // dst_line_stride,
        0);
    wait_group_events(1, &e1);

    const int sizeWeight = IC * 3 * 3;
    e1 = async_work_group_copy(w_local, w + get_group_id(1) * sizeWeight, sizeWeight, 0);
    wait_group_events(1, &e1);

    int oh = get_global_id(0);
    int oc = get_global_id(1);

    __local half *in = (__local half *)in_local + 1;

    int stride;
    int write_output = 0;
    __local half *src;

    if ((stride_x == 1) && (stride_y == 1)) {
        stride       = OW / 8;
        write_output = 1;
    }
    if ((stride_x == 2) && (stride_y == 2)) {
        stride       = OW / 4;
        write_output = 2;
    }

    for (int ow = 0; ow < stride; ow++) {
        float8 val = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int ic = 0; ic < IC; ++ic) {
            src             = (__local half *)((__local half8 *)(in + ic * IW * 3) + ow);
            __local half *k = (__local half *)(w_local + ic * 3 * 3);

            half8 aux_in00 = *((__local half8 *)src - 1);
            half8 aux_in01 = *((__local half8 *)src + 0);
            half8 aux_in02 = *((__local half8 *)src + 1);
            half8 aux_in10 = *((__local half8 *)(src + IW) - 1);
            half8 aux_in11 = *((__local half8 *)(src + IW) + 0);
            half8 aux_in12 = *((__local half8 *)(src + IW) + 1);
            half8 aux_in20 = *((__local half8 *)(src + IW * 2) - 1);
            half8 aux_in21 = *((__local half8 *)(src + IW * 2) + 0);
            half8 aux_in22 = *((__local half8 *)(src + IW * 2) + 1);

            short8 in00 = *((short8 *)&aux_in00);
            short8 in01 = *((short8 *)&aux_in01);
            short8 in02 = *((short8 *)&aux_in02);
            short8 in10 = *((short8 *)&aux_in10);
            short8 in11 = *((short8 *)&aux_in11);
            short8 in12 = *((short8 *)&aux_in12);
            short8 in20 = *((short8 *)&aux_in20);
            short8 in21 = *((short8 *)&aux_in21);
            short8 in22 = *((short8 *)&aux_in22);

            short8 aux_aux00 = __builtin_shave_cmu_alignvec_rri_short8(in00, in01, 14);
            short8 aux_aux01 = in01;
            short8 aux_aux02 = __builtin_shave_cmu_alignvec_rri_short8(in01, in02, 2);
            short8 aux_aux10 = __builtin_shave_cmu_alignvec_rri_short8(in10, in11, 14);
            short8 aux_aux11 = in11;
            short8 aux_aux12 = __builtin_shave_cmu_alignvec_rri_short8(in11, in12, 2);
            short8 aux_aux20 = __builtin_shave_cmu_alignvec_rri_short8(in20, in21, 14);
            short8 aux_aux21 = in21;
            short8 aux_aux22 = __builtin_shave_cmu_alignvec_rri_short8(in21, in22, 2);

            half8 aux00 = *((half8 *)&aux_aux00);
            half8 aux01 = *((half8 *)&aux_aux01);
            half8 aux02 = *((half8 *)&aux_aux02);
            half8 aux10 = *((half8 *)&aux_aux10);
            half8 aux11 = *((half8 *)&aux_aux11);
            half8 aux12 = *((half8 *)&aux_aux12);
            half8 aux20 = *((half8 *)&aux_aux20);
            half8 aux21 = *((half8 *)&aux_aux21);
            half8 aux22 = *((half8 *)&aux_aux22);

            half8 w00 = (half8)(*(k + 0));
            half8 w01 = (half8)(*(k + 1));
            half8 w02 = (half8)(*(k + 2));
            half8 w10 = (half8)(*(k + 3));
            half8 w11 = (half8)(*(k + 4));
            half8 w12 = (half8)(*(k + 5));
            half8 w20 = (half8)(*(k + 6));
            half8 w21 = (half8)(*(k + 7));
            half8 w22 = (half8)(*(k + 8));

            val += convert_float8(aux00) * convert_float8(w00);
            val += convert_float8(aux01) * convert_float8(w01);
            val += convert_float8(aux02) * convert_float8(w02);
            val += convert_float8(aux10) * convert_float8(w10);
            val += convert_float8(aux11) * convert_float8(w11);
            val += convert_float8(aux12) * convert_float8(w12);
            val += convert_float8(aux20) * convert_float8(w20);
            val += convert_float8(aux21) * convert_float8(w21);
            val += convert_float8(aux22) * convert_float8(w22);
        }
        if (write_output == 2) *((__local half4 *)(out_local) + ow) = convert_half4(val.s0246);
        if (write_output == 1) *((__local half8 *)(out_local) + ow) = convert_half8(val);
    }

    for (int ow = OW & ~(0x7); ow < OW; ow++) {
        float val = 0.0f;
        for (int ic = 0; ic < IC; ++ic) {
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int iw = ow * stride_x - pad_x + kx * dilation_x;
                    int ih = oh * stride_y - pad_y + ky * dilation_y;

                    val += convert_float(in[ic * IW * 3 + (ky * dilation_y) * IW + iw])
                           * convert_float(w_local[ic * 3 * 3 + ky * 3 + kx]);
                }
            }
        }
        out_local[ow] = convert_half(val);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy(
        out + get_group_id(1) * OW * OH + get_group_id(0) * OW,
        out_local,
        OW,
        0);
    wait_group_events(1, &e2);
}
