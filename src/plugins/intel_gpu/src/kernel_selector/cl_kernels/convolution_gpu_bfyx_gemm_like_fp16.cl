// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if defined(cl_intel_subgroups_short)
#define TILE_M          1
#define TILE_K          FILTER_SIZE_X
#define TILE_N          32

REQD_SUB_GROUP_SIZE(16)
KERNEL(convolution_f16)(
    const __global half *src0,
    __global half *dst,
    const __global half *src1
#if BIAS_TERM
    , const __global half *bias
#endif
)
{
#include "include/batch_headers/vec_typedefs.cl"

    const unsigned group_x = get_group_id(0);
    const unsigned group_y = get_group_id(1);
    const unsigned global_x = (uint)get_global_id(0);
    const unsigned global_y = (uint)get_global_id(1);

#if GROUPED
    const unsigned b_g = (uint)get_global_id(2);
    const unsigned global_z = b_g / FILTER_GROUPS_NUM;
    const unsigned g = b_g % FILTER_GROUPS_NUM;
#else
    const unsigned global_z = (uint)get_global_id(2);
    const unsigned g = 0;
#endif

    unsigned interleaved_y;
    unsigned kernel_y;
    unsigned kernel_idx;

    // Result ctile (*dst) is M rows x N columns
    // LWG size is 1x16.  Thus each thread calculates 16*M rows x N cols of ctile.
    half16  blockC00 = 0.f;
    half16  blockC10 = 0.f;

    const uint in_group_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
#if defined(INPUT_BUFFER_WIDTH_PADDED) && defined(INPUT_BUFFER_HEIGHT_PADDED)
    const uint src0_read_offset_const = INPUT0_OFFSET_WITH_PADDING + in_group_offset
     + INPUT0_BATCH_PITCH * global_z                                                         // batch offset
     + ( ( global_y / OUTPUT_SIZE_X ) * STRIDE_SIZE_Y * INPUT0_Y_PITCH )                     // y offset
     + ( ( global_y % OUTPUT_SIZE_X ) * STRIDE_SIZE_X );                                     // x offset
#elif !defined(INPUT_BUFFER_WIDTH_PADDED) && !defined(INPUT_BUFFER_HEIGHT_PADDED)
    #pragma error - fix this path
    const int y_offset = ( global_y / OUTPUT_SIZE_X ) * STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int x_offset = ( global_y % OUTPUT_SIZE_X ) * STRIDE_SIZE_X - PADDING_SIZE_X;
    uint src0_read_offset = INPUT_OFFSET + in_group_offset + INPUT0_BATCH_PITCH * global_z
                            + y_offset * INPUT0_Y_PITCH;

    int partial_left = 0, partial_right = 0;
    if (x_offset < 0)
    {
        partial_left = min((int) FILTER_SIZE_X, (int) abs(x_offset));
        src0_read_offset -= partial_left;
    }
    else
    {
        partial_left = 0;
        src0_read_offset +=  x_offset;
    }
    if ((x_offset + FILTER_SIZE_X) >= INPUT_SIZE_X)
        partial_right = min(FILTER_SIZE_X, INPUT_SIZE_X - x_offset);
    else
        partial_right = FILTER_SIZE_X;

#elif defined(INPUT_BUFFER_WIDTH_PADDED)
    #pragma error - fix this path
    // TODO: Handle offset
    const int y_offset = ( global_y / OUTPUT_SIZE_X ) * STRIDE_SIZE_Y -PADDING_SIZE_Y;
    int src0_read_offset = in_group_offset + INPUT0_BATCH_PITCH * global_z        // batch offset
     + y_offset * INPUT0_Y_PITCH                              // y offset
     + ( ( global_y % OUTPUT_SIZE_X ) * STRIDE_SIZE_X );                // x offset
#endif

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    uint src0_read_offset = src0_read_offset_const;
    uint src1_read_offset = ( global_x * TILE_N * 2) + g * FILTER_GROUPS_PITCH;

#define DOT_PRODUCT_16( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB,  0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB,  1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB,  2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB,  3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB,  4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB,  5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB,  6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB,  7 ), _result.s7 );  \
        _result.s8 = mad( _rowA, sub_group_broadcast( colB,  8 ), _result.s8 );  \
        _result.s9 = mad( _rowA, sub_group_broadcast( colB,  9 ), _result.s9 );  \
        _result.sa = mad( _rowA, sub_group_broadcast( colB, 10 ), _result.sa );  \
        _result.sb = mad( _rowA, sub_group_broadcast( colB, 11 ), _result.sb );  \
        _result.sc = mad( _rowA, sub_group_broadcast( colB, 12 ), _result.sc );  \
        _result.sd = mad( _rowA, sub_group_broadcast( colB, 13 ), _result.sd );  \
        _result.se = mad( _rowA, sub_group_broadcast( colB, 14 ), _result.se );  \
        _result.sf = mad( _rowA, sub_group_broadcast( colB, 15 ), _result.sf );  \
    }
    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (FILTER_SIZE_X) of each input patch
    // and FILTER_SIZE_X/2 rows of interleaved filter.
    unsigned patch_depth = 0;
    __attribute__((opencl_unroll_hint(1)))
    do
    {
        int patch_row = 0;
        __attribute__((opencl_unroll_hint(1)))
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at half16 granularity.
            // The exception is that if FILTER_SIZE_X is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for FILTER_SIZE_X=3.
            // (0, 0) (16, 0) (32, 0) (48, 0) ...     (0, 0) ( 0, 1) (16, 0) ( 0, 1) (32, 0) (0, 1) (48, 0) ...
            // (0, 1) (16, 1) (32, 1) (48, 1) ... =>  (0, 2) (16, 2) (32, 2) (48, 2) ...
            // (0, 2) (16, 2) (32, 2) (48, 2) ...     ...
            // ...
            const bool kernel_width_is_odd = FILTER_SIZE_X % 2 == 1;
            #if defined(INPUT_BUFFER_WIDTH_PADDED) && defined(INPUT_BUFFER_HEIGHT_PADDED)

            // in case the data is not aligned to sizeof(T)*FILTER_SIZE_X we need to use vload or set the data in a loop
            half blockA00[FILTER_SIZE_X];
            {
                unsigned i = 0;
                LOOP(FILTER_SIZE_X, i,
                {
#if LEFTOVERS == 1
                    if(src0_read_offset_const + (FILTER_SIZE_Y - 1) * INPUT0_Y_PITCH + (FILTER_IFM_NUM - 1) * (INPUT0_FEATURE_PITCH - ( FILTER_SIZE_Y * INPUT0_Y_PITCH )) >= INPUT0_BATCH_NUM * INPUT0_BATCH_PITCH)
                    {
                        if(src0_read_offset + i < INPUT0_BATCH_NUM * INPUT0_BATCH_PITCH)
                            blockA00[i] = src0[src0_read_offset + i];
                    }
                    else
#endif
                        blockA00[i] = src0[src0_read_offset + i];
                } )
            }

            half*  pblockA00 = (half*)(&blockA00);

            #elif !defined(INPUT_BUFFER_WIDTH_PADDED) && !defined(INPUT_BUFFER_HEIGHT_PADDED)
            // TODO: Fixed vload issue in this path.
            #pragma error
            typedef CAT( half, FILTER_SIZE_X ) half_t;
            half_t blockA00;
            half*  pblockA00 = (half*)(&blockA00);
            #if (PADDING_SIZE_X == 1) && (INPPUT_PADDING_Y == 1) && (FILTER_SIZE_X == 3) && (FILTER_SIZE_Y == 3)
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_SIZE_Y))
            {
                blockA00 = { 0 };
            }
            else
            {
                 blockA00 = src0[src0_read_offset - partial_left];
                 if (partial_left) pblockA00[0] = 0;
                 if (partial_right != FILTER_SIZE_X) pblockA00[FILTER_SIZE_X - 1] = 0;
            }
            #else
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_SIZE_Y))
            {
                blockA00 = { 0 };
            }
            else
            {
                 blockA00 = src0[src0_read_offset - partial_left];
                 for (unsigned i = 0; i < partial_left; ++i) pblockA00[i] = 0;
                 for (unsigned i = partial_right; i < FILTER_SIZE_X; ++i) pblockA00[i] = 0;

            }
            #endif
            #elif defined(INPUT_BUFFER_WIDTH_PADDED)
            // TODO: Fixed vload issue in this path.
            #pragma error
            if ((y_offset +  patch_row < 0) || ((y_offset + patch_row) >= INPUT_SIZE_Y))
            {
                blockA00 = { 0 };
            }
            else
            {
                blockA00 = src0[src0_read_offset];
            }
            #endif
            src0_read_offset += INPUT0_Y_PITCH;

            ushort blockB00[FILTER_SIZE_X * 2];
            ushort4* p4BlockB00 = (ushort4*)blockB00;
            ushort2* p2BlockB00 = (ushort2*)blockB00;
            half* pBlockB00  = (half*)blockB00;

            interleaved_y = 0;
            LOOP(FILTER_SIZE_X_DIV2, interleaved_y,
            {
                p4BlockB00[interleaved_y] = _sub_group_block_read_us4( (const __global ushort*)src1 + src1_read_offset );
                src1_read_offset += ALIGNED_OFM_PER_GROUP * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p2BlockB00[FILTER_SIZE_X - 1] = _sub_group_block_read_us2( (const __global ushort*)src1 + src1_read_offset );
                src1_read_offset += ALIGNED_OFM_PER_GROUP * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(FILTER_SIZE_X_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }
        while( ++patch_row < FILTER_SIZE_Y );

        src0_read_offset += INPUT0_FEATURE_PITCH - ( FILTER_SIZE_Y * INPUT0_Y_PITCH ); // reset to start of next slice of patch
    }
    while ( ++patch_depth < FILTER_IFM_NUM );

    #undef DOT_PRODUCT_16

    const uint out_group_offset = g * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global half *out = dst + OUTPUT_OFFSET + out_group_offset
     + global_z * OUTPUT_BATCH_PITCH                                                   // batch offset
     + ( group_x * TILE_N ) * OUTPUT_FEATURE_PITCH                                     // channel offset
     + ( ( global_y * TILE_M ) / OUTPUT_SIZE_X ) * OUTPUT_Y_PITCH                      // y offset
     + ( ( global_y * TILE_M ) % OUTPUT_SIZE_X );                                      // x offset


    if (global_y * TILE_M < OUTPUT_SIZE_X * OUTPUT_SIZE_Y )
    {
         #if BIAS_TERM
         __global half16* biasPtr = (__global half16*) (bias + group_x * TILE_N + g * FILTER_OFM_NUM);
         #endif

#if ( ( FILTER_OFM_NUM % TILE_N ) == 0 )

        #if BIAS_TERM
        blockC00 += *biasPtr;
        blockC10 += *(biasPtr + 1);
        #endif

        blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
        blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);

        for (unsigned i = 0; i < 16; i++)
        {
            out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
        }

#elif ( ( FILTER_OFM_NUM % 16 ) == 0 )
        if ( ( global_x + 1 ) < get_global_size(0) )
        {
            #if BIAS_TERM
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
            blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);

            for ( unsigned i = 0; i < 16; i++ )
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
            }
        }
        else
        {
            #if BIAS_TERM
            blockC00 += *biasPtr;
            #endif

            blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);

            for (unsigned i = 0; i < 16; i++)
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            }
        }
#else
        if ( ( global_x + 1 ) < get_global_size(0) )
        {
            #if BIAS_TERM
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
            blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);

            for ( unsigned i = 0; i < 16; i++ )
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
            }
        }
        else
        {
#if ( (FILTER_OFM_NUM % TILE_N) > 16 )

            #if BIAS_TERM
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            #endif

            blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
            blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);

            for (unsigned i = 0; i < 16 ; i++)
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            }
            for (unsigned i = 0; i < FILTER_OFM_NUM % 16 ; i++)
            {
                out[(16+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
            }
#else
            #if BIAS_TERM
            blockC00 += *biasPtr;
            #endif

            blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);

            for (unsigned i = 0; i < FILTER_OFM_NUM % 16 ; i++)
            {
                out[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
            }
#endif
        }
#endif
    }

}
#endif
