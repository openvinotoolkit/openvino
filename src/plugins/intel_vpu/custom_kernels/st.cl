// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

#define MAX_WIDTH 512

__attribute__((noinline)) void calcInd(
    __global const half *restrict theta,
    __local half *restrict weight,
    __local int *restrict ind,
    int y,
    int H,
    int x0,
    int length,
    int step,
    int W)
{
    float a = (float)y * 1.0f / H * 2 - 1;

    int x = 0;

    float8 va  = (float8){a, a, a, a, a, a, a, a};
    float8 vxy = (float8){x0 + 0, x0 + 1, x0 + 2, x0 + 3, x0 + 4, x0 + 5, x0 + 6, x0 + 7};

    for (; x <= length - 8; x += 8, vxy += 8) {
        float8 va1 = vxy * 1.0f / W * 2 - 1.f;

        float8 vx = (va * theta[0] + va1 * theta[1] + theta[2] + 1.f) / 2.f * H;
        float8 vy = (va * theta[3] + va1 * theta[4] + theta[5] + 1.f) / 2.f * W;

        const int8 ix = convert_int8(vx) - ((vx < 0) & 1);
        const int8 iy = convert_int8(vy) - ((vy < 0) & 1);

        float8 ax = vx - convert_float8(ix);
        float8 ay = vy - convert_float8(iy);
        float8 bx = 1.f - ax;
        float8 by = 1.f - ay;

        union {
            int8 d;
            uint8 i;
        } check_x;

        check_x.d = ix;
        int8 b01  = check_x.i < (uint8)H;

        check_x.d = ix + 1;
        int8 b45  = check_x.i < (uint8)H;

        union {
            int8 d;
            uint8 i;
        } check_y;

        check_y.d = iy;
        int8 b23  = check_y.i < (uint8)W;

        check_y.d = iy + 1;
        int8 b67  = check_y.i < (uint8)W;

        int8 b0123 = b01 & b23;
        int8 b0167 = b01 & b67;
        int8 b4523 = b45 & b23;
        int8 b4567 = b45 & b67;

        int8 TL_id = ((ix + 0) * W + (iy + 0)) * (b0123 & 1);
        int8 BL_id = ((ix + 1) * W + (iy + 0)) * (b4523 & 1);
        int8 TR_id = ((ix + 0) * W + (iy + 1)) * (b0167 & 1);
        int8 BR_id = ((ix + 1) * W + (iy + 1)) * (b4567 & 1);

        union {
            float8 f;
            int8 i;
        } w0;
        w0.f = bx * by;
        union {
            float8 f;
            int8 i;
        } w1;
        w1.f = ax * by;
        union {
            float8 f;
            int8 i;
        } w2;
        w2.f = bx * ay;
        union {
            float8 f;
            int8 i;
        } w3;
        w3.f = ax * ay;

        w0.i = w0.i & b0123;
        w1.i = w1.i & b4523;
        w2.i = w2.i & b0167;
        w3.i = w3.i & b4567;

        *((__local half8 *)(weight + x + 0 * step)) = convert_half8(w0.f);
        *((__local half8 *)(weight + x + 1 * step)) = convert_half8(w1.f);
        *((__local half8 *)(weight + x + 2 * step)) = convert_half8(w2.f);
        *((__local half8 *)(weight + x + 3 * step)) = convert_half8(w3.f);

        *((__local int8 *)(ind + x + 0 * step)) = TL_id;
        *((__local int8 *)(ind + x + 1 * step)) = BL_id;
        *((__local int8 *)(ind + x + 2 * step)) = TR_id;
        *((__local int8 *)(ind + x + 3 * step)) = BR_id;
    }

    for (; x < length; x++) {
        float a1 = (float)(x0 + x) * 1.0f / W * 2 - 1;

        float fx = (a * theta[0] + a1 * theta[1] + theta[2] + 1) / 2 * H;
        float fy = (a * theta[3] + a1 * theta[4] + theta[5] + 1) / 2 * W;

        const int ix = (int)(fx) - (fx < 0);
        const int iy = (int)(fy) - (fy < 0);

        float ax = fx - ix;
        float ay = fy - iy;
        float bx = 1 - ax;
        float by = 1 - ay;

        int b0 = ix >= 0;
        int b4 = ix >= -1;
        int b1 = ix < H;
        int b5 = ix < H - 1;

        int b2 = iy >= 0;
        int b6 = iy >= -1;
        int b3 = iy < W;
        int b7 = iy < W - 1;

        int b01 = b0 & b1;
        int b23 = b2 & b3;
        int b45 = b4 & b5;
        int b67 = b6 & b7;

        int b0123 = b01 & b23;
        int b0167 = b01 & b67;
        int b4523 = b45 & b23;
        int b4567 = b45 & b67;

        int TL_id = ((ix + 0) * W + (iy + 0)) * b0123;
        int BL_id = ((ix + 1) * W + (iy + 0)) * b4523;
        int TR_id = ((ix + 0) * W + (iy + 1)) * b0167;
        int BR_id = ((ix + 1) * W + (iy + 1)) * b4567;

        half w0 = bx * by * b0123;
        half w1 = ax * by * b4523;
        half w2 = bx * ay * b0167;
        half w3 = ax * ay * b4567;

        weight[x + 0 * step] = w0;
        weight[x + 1 * step] = w1;
        weight[x + 2 * step] = w2;
        weight[x + 3 * step] = w3;

        ind[x + 0 * step] = TL_id;
        ind[x + 1 * step] = BL_id;
        ind[x + 2 * step] = TR_id;
        ind[x + 3 * step] = BR_id;
    }
}

__attribute__((noinline)) void apply(
    __global half const *restrict src,
    __local half const *restrict weight,
    __local int const *restrict ind,
    __local half *restrict dst,
    int src_stride,
    int step)
{
    int x = 0;
    for (; x <= src_stride - 8; x += 8) {
        int8 TL_id = *((__local int8 *)(ind + x + 0 * step));
        int8 BL_id = *((__local int8 *)(ind + x + 1 * step));
        int8 TR_id = *((__local int8 *)(ind + x + 2 * step));
        int8 BR_id = *((__local int8 *)(ind + x + 3 * step));

        half8 w00 = *((__local half8 *)(weight + x + 0 * step));
        half8 w01 = *((__local half8 *)(weight + x + 1 * step));
        half8 w02 = *((__local half8 *)(weight + x + 2 * step));
        half8 w03 = *((__local half8 *)(weight + x + 3 * step));

        half8 TL = (half8){
            src[TL_id[0]], src[TL_id[1]],
            src[TL_id[2]], src[TL_id[3]],
            src[TL_id[4]], src[TL_id[5]],
            src[TL_id[6]], src[TL_id[7]]};
        half8 TR = (half8){
            src[TR_id[0]], src[TR_id[1]],
            src[TR_id[2]], src[TR_id[3]],
            src[TR_id[4]], src[TR_id[5]],
            src[TR_id[6]], src[TR_id[7]]};
        half8 BL = (half8){
            src[BL_id[0]], src[BL_id[1]],
            src[BL_id[2]], src[BL_id[3]],
            src[BL_id[4]], src[BL_id[5]],
            src[BL_id[6]], src[BL_id[7]]};
        half8 BR = (half8){
            src[BR_id[0]], src[BR_id[1]],
            src[BR_id[2]], src[BR_id[3]],
            src[BR_id[4]], src[BR_id[5]],
            src[BR_id[6]], src[BR_id[7]]};

        half8 res = w00 * TL + w01 * BL + w02 * TR + w03 * BR;

        *((__local half8 *)(dst + x)) = res;
    }

    for (; x < src_stride; x++) {
        int TL_id = ind[x + 0 * step];
        int BL_id = ind[x + 1 * step];
        int TR_id = ind[x + 2 * step];
        int BR_id = ind[x + 3 * step];

        half w00 = weight[x + 0 * step];
        half w01 = weight[x + 1 * step];
        half w02 = weight[x + 2 * step];
        half w03 = weight[x + 3 * step];

        half TL = src[TL_id];
        half TR = src[TR_id];
        half BL = src[BL_id];
        half BR = src[BR_id];

        half res = w00 * TL + w01 * BL + w02 * TR + w03 * BR;

        dst[x] = res;
    }
}

__kernel void ocl_st(
    __global half const *const restrict src_data,
    __global half const *const restrict theta,
    __global half *const restrict dst_data,
    int C,
    int W)
{
    __local int ind[4 * MAX_WIDTH] __attribute__((aligned(16)));
    __local half weight[4 * MAX_WIDTH] __attribute__((aligned(16)));
    __local half local_dst[4 * 1024];

    int w = get_group_id(0);

    int y = get_global_id(1);
    int H = get_global_size(1);

    const int x0         = w * MAX_WIDTH;
    const int x1         = min(x0 + MAX_WIDTH, W);
    const int src_stride = x1 - x0;

    calcInd(theta, weight, ind, y, H, x0, src_stride, MAX_WIDTH, W);

    for (int c = 0; c < C; c++) {
        __global half const *restrict src = src_data + c * H * W;
        __local half *restrict dst = local_dst + c * get_local_size(1) * src_stride + get_local_id(1) * src_stride;

        apply(src, weight, ind, dst, src_stride, MAX_WIDTH);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e = async_work_group_copy_3D3D(
        dst_data + get_group_id(1) * get_local_size(1) * W + x0, // dst
        local_dst, // src
        src_stride, // num_elements_per_line
        get_local_size(1), // num_lines
        0, // src_line_stride
        W - src_stride, // dst_line_stride
        C, // num planes
        0, // src plane stride
        W * (get_global_size(1) - get_local_size(1)), // dst plane stride
        0);
    wait_group_events(1, &e);
}
