// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

//////////////////////////////////////////////////////////////////////////////
// Direct Convolution
#if defined(cl_intel_subgroups_short)

#define TILE_M          DY      // Height of tile in input patches (src0)
#define TILE_K          DX      // Width of tile in input patches (src0)
#define TILE_N          16      // Num filter channels per tile (src1)

#define TILE_X          12      // Width of tile loaded in input (src0)
#define TILE_Y          10      // Height of tile loaded in input (src0)

REQD_SUB_GROUP_SIZE(16)
KERNEL(convolution_f16_10x12x16)(
    const __global half *src0,
    __global half *dst,
    const __global half *src1
#if BIAS_TERM
    , const __global half *biases
#endif
)
{
#include "include/batch_headers/vec_typedefs.cl"

    const unsigned global_x = (uint)get_global_id(0);
    const unsigned global_y = (uint)get_global_id(1);
    const unsigned global_z = (uint)get_global_id(2);
    const unsigned out_fm   = global_z % ALIGNED_OFM;
    const unsigned batch_id = global_z / ALIGNED_OFM;
    const unsigned group_x = get_group_id(0);
    const unsigned max_group_x = get_num_groups(0);
    const unsigned local_z = get_local_id(2);

#if GROUPED
    const uint group_z = (uint)get_group_id(2) % (ALIGNED_OFM_PER_GROUP / TILE_N);
    const uint g = out_fm / ALIGNED_OFM_PER_GROUP;
    const uint g_out_fm = out_fm % ALIGNED_OFM_PER_GROUP;
    const uint in_group_offset = g * FILTER_IFM_NUM * INPUT0_FEATURE_PITCH;
#else
    const unsigned group_z = get_group_id(2);
    const uint g = 0;
    const uint g_out_fm = out_fm;
    const uint in_group_offset = 0;
#endif

    half blockC[TILE_M * TILE_K] = { 0 };

    uint src0_offset_tile = INPUT0_OFFSET_WITH_PADDING      // data offset
     + in_group_offset
     + batch_id * INPUT0_BATCH_PITCH                        // batch offset
     + ( global_y * TILE_M * STRIDE_SIZE_Y ) * INPUT0_Y_PITCH    // y offset
     + ( global_x * TILE_K * STRIDE_SIZE_X );                    // x offset
    uint src0_offset = src0_offset_tile
     + ( local_z / ( TILE_X / 4 ) ) * INPUT0_Y_PITCH        // y tile offset
     + ( local_z % ( TILE_X / 4 ) ) * 4;                    // x tile offset

    const __global half *src1_read = src1 + ( group_z * TILE_N % ALIGNED_OFM_PER_GROUP ) * 2 + g * FILTER_GROUPS_PITCH;

    unsigned patch_depth = 0;
    __attribute__((opencl_unroll_hint(3)))
    do
    {
        // Load atile (input) and btile (filters).
        // Kernel data is partially interleaved.  Every 2 rows are interleaved at float16 granularity.
        // The exception is that if FILTER_SIZE_X is odd the last row is not interleaved.  The non
        // interleaved row is padded with zero to ensure same size as interleaved rows. This
        // interleaving is done to increase consecutive data to fetch which reduces loads required.
        // For example, this is how the kernel data would be arranged before/after interleaving for FILTER_SIZE_X=3.
        // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
        // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
        // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
        // ...

        #if ((INPUT0_Y_PITCH) % 4) == 0
        // aligned - can ignore vload
        half4 blockA0 = *(const __global half4 *)( src0 + src0_offset );
        half4 blockA1 = *(const __global half4 *)( src0 + src0_offset + INPUT0_Y_PITCH * 5 );
        #elif ((INPUT0_Y_PITCH) % 2) == 0
        // in case the data is not aligned to sizeof(T)*4 we need to use vload or set the data in a loop
        // first one is aligned
        half4 blockA0 = *(const __global half4 *)( src0 + src0_offset );
        half4 blockA1 = vload4(0, src0 + src0_offset + INPUT0_Y_PITCH * 5 );
        #else
        half4 blockA0 = vload4(0, src0 + src0_offset );
        half4 blockA1 = vload4(0, src0 + src0_offset + INPUT0_Y_PITCH * 5 );
        #endif
        src0_offset += INPUT0_FEATURE_PITCH;

        half blockB[FILTER_SIZE_X * FILTER_SIZE_Y];
        ushort2* p2BlockB = (ushort2*)blockB;
        ushort*  pBlockB =  (ushort* )blockB;

        const bool kernel_slice_is_odd = ( FILTER_SIZE_X * FILTER_SIZE_Y ) % 2 == 1;
        unsigned interleaved_y = 0;
        LOOP(KERNEL_SLICE_DIV2, interleaved_y,
        {
            p2BlockB[interleaved_y] = _sub_group_block_read_us2( (const __global ushort*)src1_read );
            src1_read += ALIGNED_OFM_PER_GROUP * 2;
        } )
        if ( kernel_slice_is_odd )
        {
            pBlockB[FILTER_SIZE_X * FILTER_SIZE_Y - 1] = _sub_group_block_read_us( (const __global ushort*)src1_read );
            src1_read += ALIGNED_OFM_PER_GROUP * 2;
        }

#define BLOCK_A(n) ( (n < 60) \
    ? sub_group_broadcast( blockA0[(n)%4], (n)/4 ) \
    : sub_group_broadcast( blockA1[(n-60)%4], (n-60)/4 ) )

        // Perform MADs
        // Loop through all patches in tile (patch_x/y)
        // For each patch, sum values (x/y)
        unsigned patch_y=0;
        LOOP(TILE_M, patch_y,
        {
            unsigned patch_x=0;
            LOOP(TILE_K, patch_x,
            {
                unsigned tile_idx = patch_y * TILE_X * STRIDE_SIZE_Y + patch_x * STRIDE_SIZE_X;
                unsigned out_idx  = patch_y * TILE_K + patch_x;

                unsigned y=0;
                LOOP(FILTER_SIZE_Y, y,
                {
                    unsigned x=0;
                    LOOP(FILTER_SIZE_X, x,
                    {
                        unsigned offset_idx = y * TILE_X + x;
                        unsigned out_chan_idx = y * FILTER_SIZE_X + x;

                        blockC[out_idx] = mad( BLOCK_A( tile_idx + offset_idx ), blockB[out_chan_idx], blockC[out_idx] );
                    } )
                } )
            } )
        } )
    }
    while ( ++patch_depth < FILTER_IFM_NUM );

    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // TILE_K x TILE_M x SIMD.  Partial writes most likely generated if output padding used.
    // Group stores into vectors to expedite writeback.  One large write is faster than many
    // small saves. Right-most column may be smaller if output width not divisible by tile width.
    const uint out_group_offset = g * FILTER_OFM_NUM * OUTPUT_FEATURE_PITCH;
    __global half *out = dst + OUTPUT_OFFSET + out_group_offset +
     + batch_id * OUTPUT_BATCH_PITCH            // batch offset
     + g_out_fm * OUTPUT_FEATURE_PITCH              // channel offset
     + ( global_y * TILE_M ) * OUTPUT_Y_PITCH // y offset
     + ( global_x * TILE_K );                // x offset

    if ( batch_id < OUTPUT_BATCH_NUM && g_out_fm < FILTER_OFM_NUM )
    {
#if BIAS_TERM == 0
        const half bias = 0.h;
#elif BIAS_PER_OFM
        const half bias = biases[g * FILTER_OFM_NUM + g_out_fm];
#endif

        if ( OUTPUT_SIZE_X % TILE_K == 0 ||
             group_x < max_group_x - 1 )
        {
            typedef CAT( half, TILE_K ) half_t;
            for( unsigned y = 0; y < TILE_M; y++ )
            {
                if ( global_y * TILE_M + y < OUTPUT_SIZE_Y )
                {
                    half_t vBlockC;
                    half *pvBlockC = (half*)&vBlockC;
                    for (unsigned i = 0; i < TILE_K; i++)
                    {
                    #if BIAS_TERM && BIAS_PER_OUTPUT
                        const unsigned bias_index = out_fm*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + ( global_y * TILE_M + y )*OUTPUT_SIZE_X + ( global_x * TILE_K + i);
                        const half bias = biases[bias_index];
                    #endif
                        pvBlockC[i] = ACTIVATION(blockC[y * TILE_K + i] + bias, ACTIVATION_PARAMS);
                        ((__global half*)(out + y * OUTPUT_Y_PITCH))[i] = pvBlockC[i];
                    }
                    //*(__global half_t*)(out + y * OUTPUT_Y_PITCH) = vBlockC;
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
                    for (unsigned i = 0; i < RIGHT_PARTIAL_TILE_K; i++)
                    {
                    #if BIAS_TERM && BIAS_PER_OUTPUT
                        const unsigned bias_index = out_fm*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + ( global_y * TILE_M + y )*OUTPUT_SIZE_X + ( global_x * TILE_K + i);
                        const half bias = biases[bias_index];
                    #endif
                        pvBlockC[i] = ACTIVATION(blockC[y * TILE_K + i] + bias, ACTIVATION_PARAMS);
                        ((__global half*)(out + y * OUTPUT_Y_PITCH))[i] = pvBlockC[i];
                    }
                    //*(__global half_t*)(out + y * OUTPUT_Y_PITCH) = vBlockC;
                }
            }
        }
    }
}
#endif // cl_intel_subgroups_short
