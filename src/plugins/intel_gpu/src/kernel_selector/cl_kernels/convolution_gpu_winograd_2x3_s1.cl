// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_shuffle.cl"

// --------------------------------------------------------------------------------------------------------------------------------
// L3_SIMD_4x8
// Input matrices dimensions: M x K x N
// Output matrix dimensions: M x N
// --------------------------------------------------------------------------------------------------------------------------------
#define VEC_SIZE        4   // dx
#define TILE_M          8   // dy
#define TILE_K          32
#define TILE_N          32

#define WINOGRAD_TILE_WIDTH 4
#define WINOGRAD_FILTER_HEIGHT 3
#define WINOGRAD_OUTPUT_TILE_WIDTH 2 //width of the winograd tile when transformed back to standard domain, do not confuse with outpout of this kernel (which is still in winograd domain)

#define _CAT(a,b) a##b
#define CAT(a,b) _CAT(a,b)
#define UNIT_TYPE_4 CAT(UNIT_TYPE, 4)

#define INPUT0_PITCH_SIZE_Y INPUT0_FEATURE_NUM
#define WEIGHTS_PITCH_FEATURE OUTPUT_FEATURE_NUM
#define INPUT0_PITCH_FEATURE 1

__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL(convolution_gpu_winograd_2x3_s1)
(
    const __global UNIT_TYPE *signalw,
          __global UNIT_TYPE *outputw,
    const __global UNIT_TYPE *filterw)
{
    const int INPUT0_SIZE_Y_PITCH_UNIT_4 = INPUT0_PITCH_SIZE_Y / VEC_SIZE; //for bxyf -> INPUT0_PITCH_SIZE_Y is equal to input features count, since ifm % 32 == 0, division by VEC_SIZE is ok
    const int OUTPUT_SIZE_Y_PITCH_UNIT_4 = OUTPUT_Y_PITCH / VEC_SIZE; //for bxyf -> OUTPUT_Y_PITCH is equal to output features count, since ofm % 32 == 0, division by VEC_SIZE is ok
	  const int WEIGHTS_FEATURE_PITCH_UNIT_4 = WEIGHTS_PITCH_FEATURE / VEC_SIZE; //for xyio -> WEIGHTS_PITCH_FEATURE is equal to the output features count

    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int group_z = get_group_id(2);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int local_z = get_local_id(2);

    const int no_of_tiles_x = INPUT0_SIZE_WINOGRAD_X / WINOGRAD_TILE_WIDTH;
    const int no_of_tiles_y = INPUT0_SIZE_WINOGRAD_Y - WINOGRAD_FILTER_HEIGHT + 1;

    const int x_offset_from_z_id = group_z % WINOGRAD_TILE_WIDTH;
    const int batch_idx = group_z / WINOGRAD_TILE_WIDTH;

    //y-dim size is equal to a flattened number of tiles in x-y dims,
    //since one work group processes TILE_M tiles, flattened tile idx is group_y * TILE_M,
    //this idx is then deflattened to idx in x and y dim by dividing by no_of_tiles_y,
    //note: we do not add local id because group size in y-dim is 1
    const int linear_x = (group_y * TILE_M) / no_of_tiles_y;
    const int tile_idx_y = (group_y * TILE_M) % no_of_tiles_y;
    const int x_idx = linear_x + x_offset_from_z_id * no_of_tiles_x;
    const int y_idx = tile_idx_y; //winograd tile height == 1
    const int f_idx = group_x * TILE_N + local_x * VEC_SIZE;
    const int b_idx = batch_idx;

	  const int in_tile_idx = (x_idx % WINOGRAD_TILE_WIDTH);
	  const int tile_idx_x = (x_idx / WINOGRAD_TILE_WIDTH);

    // Result ctile is M rows x N columns
    // M = 8, we have 1 rows of work-items, so we need 8/1 = 8 results down
    // N = 32, we have 8 columns of work-items, so we need 32/8 = 4 results across = 1 float4s across

    UNIT_TYPE_4 c0 = (UNIT_TYPE_4)(0.f);
    UNIT_TYPE_4 c1 = (UNIT_TYPE_4)(0.f);
    UNIT_TYPE_4 c2 = (UNIT_TYPE_4)(0.f);
    UNIT_TYPE_4 c3 = (UNIT_TYPE_4)(0.f);
    UNIT_TYPE_4 c4 = (UNIT_TYPE_4)(0.f);
    UNIT_TYPE_4 c5 = (UNIT_TYPE_4)(0.f);
    UNIT_TYPE_4 c6 = (UNIT_TYPE_4)(0.f);
    UNIT_TYPE_4 c7 = (UNIT_TYPE_4)(0.f);

    //optimal format is bxyf
    const int output_idx = b_idx * OUTPUT_BATCH_PITCH +
                           f_idx * OUTPUT_FEATURE_PITCH +
                           x_idx * OUTPUT_X_PITCH +
                           y_idx * OUTPUT_Y_PITCH;

    __global UNIT_TYPE_4 *dst = (__global UNIT_TYPE_4 *)(outputw + output_idx);

    // Src0 is used directly as atile.
    // It starts at the left side of signalw and walks across.
    // atile is M rows x K columns.
    // M = 8, we have 1 rows of work-items, so we need 8/1 = 8 rows.
    // K = 32, we have 8 columns of work-items, so we need 32/8 = 4 floats across = 1 float4s across
    const int src0_idx = local_x * VEC_SIZE * INPUT0_PITCH_FEATURE
                         + y_idx * INPUT0_FEATURE_NUM
                         + x_idx * INPUT0_SIZE_WINOGRAD_Y * INPUT0_FEATURE_NUM
                         + batch_idx * INPUT0_SIZE_WINOGRAD_X * INPUT0_SIZE_WINOGRAD_Y * INPUT0_FEATURE_NUM;

    const __global UNIT_TYPE_4 *src0 = (__global UNIT_TYPE_4 *)(signalw + src0_idx);

    // Src1 is directly used as btile.
    // It starts at the top of filterw and walks down.
    // btile is K rows x N columns.
    // K = 32, we'll process four rows at a time
    // N = 32, we have 8 columns of work-items, so we need 32/8 = 4 floats across = 1 float4s across
    const int src1_idx = local_x * VEC_SIZE
                         + (group_x * TILE_N)
                         + in_tile_idx * WINOGRAD_FILTER_HEIGHT * INPUT0_FEATURE_NUM * OUTPUT_FEATURE_NUM;

    const __global UNIT_TYPE_4 *src1 = (__global UNIT_TYPE_4 *)(filterw + src1_idx);

    UNIT_TYPE_4 a;

    // Walk ACROSS signalw and DOWN filterw:
    for (int w = 0; w < K; w += TILE_K)
    {
		//in one iteration load tile 1-width, 8-height, 4-depth (REQ: in_y % 8 == 0),
		//SIMD reads are chained along f-axis, resulting in a 1-width, 8-height, 4*8=32-depth input block (REQ: ifm % 32 == 0)
		//consecutive blocks are also chained along f-axis and overflows to y-axis, reading in total 3*f values (i.e., read all in-depth values from 3 consecutive y values and constant x)
        const UNIT_TYPE_4 a0 = src0[0 * INPUT0_SIZE_Y_PITCH_UNIT_4];
        const UNIT_TYPE_4 a1 = src0[1 * INPUT0_SIZE_Y_PITCH_UNIT_4];
        const UNIT_TYPE_4 a2 = src0[2 * INPUT0_SIZE_Y_PITCH_UNIT_4];
        const UNIT_TYPE_4 a3 = src0[3 * INPUT0_SIZE_Y_PITCH_UNIT_4];
        const UNIT_TYPE_4 a4 = src0[4 * INPUT0_SIZE_Y_PITCH_UNIT_4];
        const UNIT_TYPE_4 a5 = src0[5 * INPUT0_SIZE_Y_PITCH_UNIT_4];
        const UNIT_TYPE_4 a6 = src0[6 * INPUT0_SIZE_Y_PITCH_UNIT_4];
        const UNIT_TYPE_4 a7 = src0[7 * INPUT0_SIZE_Y_PITCH_UNIT_4];

#define DOT_PRODUCT( _i, _j ) { a = _sub_group_shuffle(a ## _i, _j); c ## _i = mad(a.x, b0, mad(a.y, b1, mad(a.z, b2, mad(a.w, b3, c ## _i)))); }

		//in one iteration load weights tile 1-width, 1-height, 4-depth from 4 different filters (ofms)
		//SIMD reads are chained along b-axis (different ofms), resulting in 1-width, 1-height, 4-depth blocks from 4*8=32 different filters
		//consecutive reads are chained along f-dim and overflows to y-dim, reading in total
#define ITERATION( _j ) \
        {   \
            const UNIT_TYPE_4 b0 = src1[0]; src1 += WEIGHTS_FEATURE_PITCH_UNIT_4; \
            const UNIT_TYPE_4 b1 = src1[0]; src1 += WEIGHTS_FEATURE_PITCH_UNIT_4; \
            const UNIT_TYPE_4 b2 = src1[0]; src1 += WEIGHTS_FEATURE_PITCH_UNIT_4; \
            const UNIT_TYPE_4 b3 = src1[0]; src1 += WEIGHTS_FEATURE_PITCH_UNIT_4; \
            \
            DOT_PRODUCT(0, _j) \
            DOT_PRODUCT(1, _j) \
            DOT_PRODUCT(2, _j) \
            DOT_PRODUCT(3, _j) \
            DOT_PRODUCT(4, _j) \
            DOT_PRODUCT(5, _j) \
            DOT_PRODUCT(6, _j) \
            DOT_PRODUCT(7, _j) \
        }

        // If I had #pragma unroll I wouldn't need to do this manually...

        // We need K/VEC_SIZE iterations.
        // K = 32, VEC_SIZE = 4
        // So, 32/4 = 8 iterations.
        ITERATION(0);
        ITERATION(1);
        ITERATION(2);
        ITERATION(3);
        ITERATION(4);
        ITERATION(5);
        ITERATION(6);
        ITERATION(7);

#undef ITERATION
#undef DOT_PRODUCT

        src0 += TILE_K / VEC_SIZE;
    }

    dst[0] = c0; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
    dst[0] = c1; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
    dst[0] = c2; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
    dst[0] = c3; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
    dst[0] = c4; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
    dst[0] = c5; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
    dst[0] = c6; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
    dst[0] = c7; dst += OUTPUT_SIZE_Y_PITCH_UNIT_4;
};
