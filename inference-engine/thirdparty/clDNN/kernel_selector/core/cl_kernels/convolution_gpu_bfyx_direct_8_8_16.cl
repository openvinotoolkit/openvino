/*
// Copyright (c) 2016 Intel Corporation
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
*/

//#include "include/cnn_common.cl"

//////////////////////////////////////////////////////////////////////////////
// Direct Convolution
#if defined(cl_intel_subgroups_short)

#define TILE_M          DY      // Height of tile in input patches (src0)
#define TILE_K          DX      // Width of tile in input patches (src0)
#define TILE_N          16      // Num filter channels per tile (src1)

#define TILE_X          8       // Width of tile loaded in input (src0)
#define TILE_Y          8       // Height of tile loaded in input (src0)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void convolution_f16_8x8x16(
    const __global half *src0,
    __global half *dst,
    const __global half *src1,
    const __global half *biases)
{
    const unsigned global_x = (uint)get_global_id(0);
    const unsigned global_y = (uint)get_global_id(1);
    const unsigned global_z = (uint)get_global_id(2);
    const unsigned out_fm   = global_z % ALIGNED_OFM;
    const unsigned batch_id = global_z / ALIGNED_OFM;
    const unsigned group_x = get_group_id(0);
    const unsigned group_z = get_group_id(2);
    const unsigned max_group_x = get_num_groups(0);
    const unsigned local_z = get_local_id(2);

    half blockC[TILE_M * TILE_K] = { 0 };

    uint src0_offset_tile =
       batch_id * INPUT_BATCH_PITCH                         // batch offset
     + ( global_y * TILE_M * STRIDE_Y ) * INPUT_Y_PITCH   // y offset
     + ( global_x * TILE_K * STRIDE_X );                    // x offset
    uint src0_offset = src0_offset_tile
     + ( local_z / ( TILE_X / 4 ) ) * INPUT_Y_PITCH       // y tile offset
     + ( local_z % ( TILE_X / 4 ) ) * 4;                    // x tile offset

    const __global half *src1_read = src1 + ( group_z * TILE_N % ALIGNED_OFM ) * 2;

    unsigned patch_depth = 0;
    __attribute__((opencl_unroll_hint(3)))
    do
    {
        // Load atile (input) and btile (filters).
        // Kernel data is partially interleaved.  Every 2 rows are interleaved at float16 granularity.
        // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
        // interleaved row is padded with zero to ensure same size as interleaved rows. This
        // interleaving is done to increase consecutive data to fetch which reduces loads required.
        // For example, this is how the kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
        // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
        // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
        // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
        // ...
        
        // in case the data is not aligned to sizeof(T)*KERNEL_WIDTH we need to use vload or set the data in a loop
        half4 blockA = vload4(0, src0 + src0_offset );
        src0_offset += INPUT_FEATURE_PITCH;

        half blockB[KERNEL_WIDTH * KERNEL_HEIGHT];
        ushort2* p2BlockB = (ushort2*)blockB;
        ushort*  pBlockB =  (ushort* )blockB;

        const bool kernel_slice_is_odd = ( KERNEL_WIDTH * KERNEL_HEIGHT ) % 2 == 1;
        unsigned interleaved_y = 0;
        LOOP(KERNEL_SLICE_DIV2, interleaved_y,
        {
            p2BlockB[interleaved_y] = intel_sub_group_block_read_us2( (const __global ushort*)src1_read );
            src1_read += ALIGNED_OFM * 2;
        } )
        if ( kernel_slice_is_odd )
        {
            pBlockB[KERNEL_WIDTH * KERNEL_HEIGHT - 1] = intel_sub_group_block_read_us( (const __global ushort*)src1_read );
            src1_read += ALIGNED_OFM * 2;
        }

#define BLOCK_A(n) sub_group_broadcast( blockA[(n)%4], (n)/4 )

        // Perform MADs
        // Loop through all patches in tile (patch_x/y)
        // For each patch, sum values (x/y)
        unsigned patch_y=0;
        LOOP(TILE_M, patch_y,
        {
            unsigned patch_x=0;
            LOOP(TILE_K, patch_x,
            {
                unsigned tile_idx = patch_y * TILE_X * STRIDE_Y + patch_x * STRIDE_X;
                unsigned out_idx  = patch_y * TILE_K + patch_x;

                unsigned y=0;
                LOOP(KERNEL_HEIGHT, y,
                {
                    unsigned x=0;
                    LOOP(KERNEL_WIDTH, x,
                    {
                        unsigned offset_idx = y * TILE_X + x;
                        unsigned out_chan_idx = y * KERNEL_WIDTH + x;

                        blockC[out_idx] = mad( BLOCK_A( tile_idx + offset_idx ), blockB[out_chan_idx], blockC[out_idx] );
                    } )
                } )
            } )
        } )
    }
    while ( ++patch_depth < INPUT_FEATURE_NUM );

    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // TILE_K x TILE_M x SIMD.  Partial writes most likely generated if output padding used.
    // Group stores into vectors to expedite writeback.  One large write is faster than many
    // small saves. Right-most column may be smaller if output width not divisible by tile width.
    __global half *out = dst
     + batch_id * OUTPUT_BATCH_PITCH            // batch offset
     + out_fm * OUTPUT_FEATURE_PITCH              // channel offset
     + ( global_y * TILE_M ) * OUTPUT_Y_PITCH // y offset
     + ( global_x * TILE_K );                // x offset

    if ( batch_id < OUTPUT_BATCH_NUM && out_fm < OUTPUT_FEATURE_NUM )
    {
        half bias = biases[out_fm];
        if ( OUTPUT_SIZE_X % TILE_K == 0 ||
             group_x < max_group_x - 1 )
        {
            typedef CAT( half, TILE_K ) half_t;
            half bias = biases[out_fm];
            for( unsigned y = 0; y < TILE_M; y++ )
            {
                if ( global_y * TILE_M + y < OUTPUT_SIZE_Y )
                {
                    half_t vBlockC;
                    half *pvBlockC = (half*)&vBlockC;
                    for (unsigned i = 0; i < TILE_K; i++) pvBlockC[i] = activation_function(blockC[y * TILE_K + i] + bias, ACTIVATION_PARAMS);
                    *(__global half_t*)(out + y * OUTPUT_Y_PITCH) = vBlockC;
                }
            }
        }
        else
        {
            typedef CAT( half, RIGHT_PARTIAL_TILE_K ) half_t;
            for( unsigned y = 0; y < TILE_M; y++ )
            {
                if ( global_y * TILE_M + y < OUTPUT_SIZE_Y )
                {
                    half_t vBlockC;
                    half *pvBlockC = (half*)&vBlockC;
                    for (unsigned i = 0; i < RIGHT_PARTIAL_TILE_K; i++) pvBlockC[i] = activation_function(blockC[y * TILE_K + i] + bias, ACTIVATION_PARAMS);
                    *(__global half_t*)(out + y * OUTPUT_Y_PITCH) = vBlockC;
                }
            }
        }
    }
}
#endif // cl_intel_subgroups_short
