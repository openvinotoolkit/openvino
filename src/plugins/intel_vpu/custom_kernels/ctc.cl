// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__global half *find(__global const half *begin, __global const half *end, half value)
{
    while (begin != end) {
        if (*begin == value) {
            return begin;
        }
        ++begin;
    }
    return end;
}

__kernel void CTCDecoder(
    __global half *restrict probabilities,
    __global half *restrict sequence_indicators,
    __global half *restrict output,
    int width,
    int height,
    int channels)
{
    __local half local_src[88 * 1 * 77];
    __local half local_dst[88 * 1];

    event_t e1 = async_work_group_copy_2D2D(
        local_src, // dst
        probabilities, // src
        width, // num_elements_per_line,
        height * channels, // num_lines,
        width * (height - 1), // src_line_stride,
        width * (height - 1), // dst_line_stride,
        0);

    wait_group_events(1, &e1);

    const int T = channels; // Time
    const int B = height; // Batches
    const int C = width; // Chars

    #pragma unroll 4
    for (int i = 0; i < B * T; i++) {
        local_dst[i] = -1.h;
    }

    int output_index = 0;

    for (int b = 0; b < B; ++b) {
        __global const half *restrict seq_ind = sequence_indicators + b * T;
        const int seq_len = find(seq_ind + 1, seq_ind + T, 0.h) - seq_ind;
        const int time    = min(seq_len, T);

        int prev_class_idx = -1;

        #pragma unroll 4
        for (int t = 0; t < time; ++t) {
            __local const half *restrict probs = local_src + b * C + t * C * B;

            int max_class_idx = 0;
            half max_prob     = probs[0];
            for (int c = 1; c < C; ++c) {
                const half prob = probs[c];
                if (prob > max_prob) {
                    max_class_idx = c;
                    max_prob      = prob;
                }
            }

            if (max_class_idx < C - 1 && max_class_idx != prev_class_idx) {
                local_dst[b * T + output_index] = (half)max_class_idx;
                output_index++;
            }

            prev_class_idx = max_class_idx;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy_2D2D(
        output, // dst
        local_dst, // src
        channels, // num_elements_per_line,
        height, // num_lines,
        0, // src_line_stride,
        0, // dst_line_stride,
        0);

    wait_group_events(1, &e2);
}
