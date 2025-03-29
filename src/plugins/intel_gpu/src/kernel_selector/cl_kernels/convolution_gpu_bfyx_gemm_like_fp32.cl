// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/sub_group.cl"

#define TILE_M          2
#define TILE_K          FILTER_SIZE_X
#define TILE_N          32

REQD_SUB_GROUP_SIZE(8)
KERNEL(convolution_f32)(
    const __global float *src0,
    __global float *dst,
    const __global float *src1
#if BIAS_TERM
    , const __global float *bias
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
    // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
    float8  blockC00 = 0.f;
    float8  blockC10 = 0.f;
    float8  blockC20 = 0.f;
    float8  blockC30 = 0.f;
    float8  blockC01 = 0.f;
    float8  blockC11 = 0.f;
    float8  blockC21 = 0.f;
    float8  blockC31 = 0.f;

    const uint in_group_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
    const uint src0_read_offset0_const = INPUT0_OFFSET_WITH_PADDING + in_group_offset
     + INPUT0_BATCH_PITCH * global_z                                                         // batch offset
     + ( ( ( global_y * TILE_M + 0 ) / OUTPUT_SIZE_X ) * STRIDE_SIZE_Y * INPUT0_Y_PITCH )    // y offset
     + ( ( ( global_y * TILE_M + 0 ) % OUTPUT_SIZE_X ) * STRIDE_SIZE_X );                    // x offset
    const uint src0_read_offset1_const = INPUT0_OFFSET_WITH_PADDING + in_group_offset
     + INPUT0_BATCH_PITCH * global_z                                                 // batch offset
     + ( ( ( global_y * TILE_M + 1 ) / OUTPUT_SIZE_X ) * STRIDE_SIZE_Y * INPUT0_Y_PITCH )    // y offset
     + ( ( ( global_y * TILE_M + 1 ) % OUTPUT_SIZE_X ) * STRIDE_SIZE_X );                    // x offset

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    uint src0_read_offset0 = src0_read_offset0_const;
    uint src0_read_offset1 = src0_read_offset1_const;
    uint src1_read_offset = ( global_x * TILE_N * 2) + g * FILTER_GROUPS_PITCH;

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB,  0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB,  1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB,  2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB,  3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB,  4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB,  5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB,  6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB,  7 ), _result.s7 );  \
    }

    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (FILTER_SIZE_X) of each input patch
    // and FILTER_SIZE_X/2 rows of interleaved filter.
    unsigned patch_depth = 0;
    do
    {
        unsigned patch_row = 0;
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at float8 granularity.
            // The exception is that if FILTER_SIZE_X is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for FILTER_SIZE_X=3.
            // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
            // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
            // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
            // ...
            const bool kernel_width_is_odd = FILTER_SIZE_X % 2 == 1;

            float blockA00[FILTER_SIZE_X];
            float blockA01[FILTER_SIZE_X];

            // in case the data is not aligned to sizeof(T)*FILTER_SIZE_X we need to use vload or set the data in a loop
            {
                unsigned i = 0;
                LOOP(FILTER_SIZE_X, i,
                {
#if LEFTOVERS == 1
                    if(src0_read_offset0_const + (FILTER_SIZE_Y - 1) * INPUT0_Y_PITCH + (FILTER_IFM_NUM - 1) * (INPUT0_FEATURE_PITCH - ( FILTER_SIZE_Y * INPUT0_Y_PITCH )) >= INPUT0_BATCH_NUM * INPUT0_BATCH_PITCH)
                    {
                        if(src0_read_offset0 + i < INPUT0_BATCH_NUM * INPUT0_BATCH_PITCH)
                            blockA00[i] = src0[src0_read_offset0 + i];
                    }
                    else
#endif
                        blockA00[i] = src0[src0_read_offset0 + i];

#if LEFTOVERS == 1
                    if(src0_read_offset1_const + (FILTER_SIZE_Y - 1) * INPUT0_Y_PITCH + (FILTER_IFM_NUM - 1) * (INPUT0_FEATURE_PITCH - ( FILTER_SIZE_Y * INPUT0_Y_PITCH )) >= INPUT0_BATCH_NUM * INPUT0_BATCH_PITCH)
                    {
                        if(src0_read_offset1 + i < INPUT0_BATCH_NUM * INPUT0_BATCH_PITCH)
                            blockA01[i] = src0[src0_read_offset1 + i];
                    }
                    else
#endif
                        blockA01[i] = src0[src0_read_offset1 + i];
                } )
            }

            float*  pblockA00 = (float*)(&blockA00);
            float*  pblockA01 = (float*)(&blockA01);

            src0_read_offset0 += INPUT0_Y_PITCH;
            src0_read_offset1 += INPUT0_Y_PITCH;


            float blockB00[FILTER_SIZE_X*4];
            float8* p8BlockB00 = (float8*)blockB00;
            float4* p4BlockB00 = (float4*)blockB00;
            float*  pBlockB00 =  (float* )blockB00;

            interleaved_y = 0;
            LOOP(FILTER_SIZE_X_DIV2, interleaved_y,
            {
                p8BlockB00[interleaved_y] = as_float8( _sub_group_block_read8( (const __global uint*)src1 + src1_read_offset ) );
                src1_read_offset += ALIGNED_OFM_PER_GROUP * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p4BlockB00[FILTER_SIZE_X - 1] = as_float4( _sub_group_block_read4( (const __global uint*)src1 + src1_read_offset ) );
                src1_read_offset += ALIGNED_OFM_PER_GROUP * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(FILTER_SIZE_X_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }

        //while( ++patch_row < 1 ); //debug
        while( ++patch_row < FILTER_SIZE_Y );

        src0_read_offset0 += INPUT0_FEATURE_PITCH - ( FILTER_SIZE_Y * INPUT0_Y_PITCH ); // reset to start of next slice of patch
        src0_read_offset1 += INPUT0_FEATURE_PITCH - ( FILTER_SIZE_Y * INPUT0_Y_PITCH ); // reset to start of next slice of patch
    }
    //while ( ++patch_depth < 1 );  //debug
    while ( ++patch_depth < FILTER_IFM_NUM );

    const uint out_group_offset = g * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global float *out0 = dst + OUTPUT_OFFSET + out_group_offset
     + global_z * OUTPUT_BATCH_PITCH                                                   // batch offset
     + ( group_x * TILE_N ) * OUTPUT_FEATURE_PITCH                                     // channel offset
     + ( ( global_y * TILE_M ) / OUTPUT_SIZE_X ) * OUTPUT_Y_PITCH                      // y offset
     + ( ( global_y * TILE_M ) % OUTPUT_SIZE_X );                                      // x offset
    __global float *out1 = dst + OUTPUT_OFFSET + out_group_offset
     + global_z * OUTPUT_BATCH_PITCH                                                   // batch offset
     + ( group_x * TILE_N ) * OUTPUT_FEATURE_PITCH                                     // channel offset
     + ( ( global_y * TILE_M + 1 ) / OUTPUT_SIZE_X ) * OUTPUT_Y_PITCH                  // y offset
     + ( ( global_y * TILE_M + 1 ) % OUTPUT_SIZE_X );                                  // x offset

    #if BIAS_TERM
    __global float8* biasPtr = (__global float8*) (bias + group_x * TILE_N + g * FILTER_OFM_NUM);
    #endif

    if( global_y * TILE_M < OUTPUT_SIZE_X * OUTPUT_SIZE_Y )
    {
        if ( ( FILTER_OFM_NUM % TILE_N ) == 0 )
        {
            #if BIAS_TERM
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            blockC20 += *(biasPtr + 2);
            blockC30 += *(biasPtr + 3);
            #endif

            blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
            blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);
            blockC20 = ACTIVATION(blockC20, ACTIVATION_PARAMS);
            blockC30 = ACTIVATION(blockC30, ACTIVATION_PARAMS);

            for( unsigned i = 0; i < 8; i++ )
            {
                out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                out0[(16+i) * OUTPUT_FEATURE_PITCH] = blockC20[i];
                out0[(24+i) * OUTPUT_FEATURE_PITCH] = blockC30[i];
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                #if BIAS_TERM
                blockC00 += *biasPtr;
                blockC10 += *(biasPtr + 1);
                blockC20 += *(biasPtr + 2);
                blockC30 += *(biasPtr + 3);
                #endif

                blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
                blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);
                blockC20 = ACTIVATION(blockC20, ACTIVATION_PARAMS);
                blockC30 = ACTIVATION(blockC30, ACTIVATION_PARAMS);

                for ( unsigned i = 0; i < 8; i++ )
                {
                    out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                    out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                    out0[(16+i) * OUTPUT_FEATURE_PITCH] = blockC20[i];
                    out0[(24+i) * OUTPUT_FEATURE_PITCH] = blockC30[i];
                }
            }
            else
            {
                if ( ( FILTER_OFM_NUM % TILE_N ) >= 24 )
                {
                    #if BIAS_TERM
                    blockC00 += *biasPtr;
                    blockC10 += *(biasPtr + 1);
                    blockC20 += *(biasPtr + 2);
                    if (( FILTER_OFM_NUM % TILE_N) > 24 ) blockC30 += *(biasPtr + 3);
                    #endif

                    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
                    blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);
                    blockC20 = ACTIVATION(blockC20, ACTIVATION_PARAMS);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                        out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                        out0[(16+i) * OUTPUT_FEATURE_PITCH] = blockC20[i];
                    }

                    // remaining output channels
                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out0[(24+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC30[i], ACTIVATION_PARAMS);
                    }
                }
                else if ( ( FILTER_OFM_NUM % TILE_N ) >= 16 )
                {
                    #if BIAS_TERM
                    blockC00 += *biasPtr;
                    blockC10 += *(biasPtr + 1);
                    if (( FILTER_OFM_NUM % TILE_N) > 16 )
                        blockC20 += *(biasPtr + 2);
                    #endif

                    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
                    blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                        out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                    }

                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out0[(16+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC20[i], ACTIVATION_PARAMS);

                    }
                }
                else if ( ( FILTER_OFM_NUM % TILE_N ) >= 8 )
                {
                    #if BIAS_TERM
                    blockC00 += *biasPtr;
                    if (( FILTER_OFM_NUM % TILE_N) > 8 )
                        blockC10 += *(biasPtr + 1);
                    #endif

                    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                    }

                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out0[(8+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC10[i], ACTIVATION_PARAMS);
                    }
                }
                else
                {
                    #if BIAS_TERM
                    blockC00 += *biasPtr;
                    #endif
                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC00[i], ACTIVATION_PARAMS);
                    }
                }
            }
        }
    }

    if ((global_y * TILE_M + 1) < OUTPUT_SIZE_X * OUTPUT_SIZE_Y )
    {
        if ( ( FILTER_OFM_NUM % TILE_N ) == 0 )
        {
            #if BIAS_TERM
            blockC01 += *biasPtr;
            blockC11 += *(biasPtr + 1);
            blockC21 += *(biasPtr + 2);
            blockC31 += *(biasPtr + 3);
            #endif

            blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);
            blockC11 = ACTIVATION(blockC11, ACTIVATION_PARAMS);
            blockC21 = ACTIVATION(blockC21, ACTIVATION_PARAMS);
            blockC31 = ACTIVATION(blockC31, ACTIVATION_PARAMS);

            for( unsigned i = 0; i < 8; i++ )
            {
                out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                out1[(16+i) * OUTPUT_FEATURE_PITCH] = blockC21[i];
                out1[(24+i) * OUTPUT_FEATURE_PITCH] = blockC31[i];
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                #if BIAS_TERM
                blockC01 += *biasPtr;
                blockC11 += *(biasPtr + 1);
                blockC21 += *(biasPtr + 2);
                blockC31 += *(biasPtr + 3);
                #endif

                blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);
                blockC11 = ACTIVATION(blockC11, ACTIVATION_PARAMS);
                blockC21 = ACTIVATION(blockC21, ACTIVATION_PARAMS);
                blockC31 = ACTIVATION(blockC31, ACTIVATION_PARAMS);

                for ( unsigned i = 0; i < 8; i++ )
                {
                    out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                    out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                    out1[(16+i) * OUTPUT_FEATURE_PITCH] = blockC21[i];
                    out1[(24+i) * OUTPUT_FEATURE_PITCH] = blockC31[i];
                }
            }
            else
            {
                if ( ( FILTER_OFM_NUM % TILE_N ) >= 24 )
                {
                    #if BIAS_TERM
                    blockC01 += *biasPtr;
                    blockC11 += *(biasPtr + 1);
                    blockC21 += *(biasPtr + 2);
                    if ( ( FILTER_OFM_NUM % TILE_N ) > 24 ) blockC31 += *(biasPtr + 3);
                    #endif

                    blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);
                    blockC11 = ACTIVATION(blockC11, ACTIVATION_PARAMS);
                    blockC21 = ACTIVATION(blockC21, ACTIVATION_PARAMS);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                        out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                        out1[(16+i) * OUTPUT_FEATURE_PITCH] = blockC21[i];
                    }

                    // Remaining channels
                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out1[(24+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC31[i], ACTIVATION_PARAMS);
                    }
                }
                else if ( ( FILTER_OFM_NUM % TILE_N ) >= 16 )
                {
                    #if BIAS_TERM
                    blockC01 += *biasPtr;
                    blockC11 += *(biasPtr + 1);
                    if ( ( FILTER_OFM_NUM % TILE_N ) > 16 ) blockC21 += *(biasPtr + 2);
                    #endif

                    blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);
                    blockC11 = ACTIVATION(blockC11, ACTIVATION_PARAMS);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                        out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                    }

                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out1[(16+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC21[i], ACTIVATION_PARAMS);
                    }
                }
                else if ( ( FILTER_OFM_NUM % TILE_N ) >= 8 )
                {
                    #if BIAS_TERM
                    blockC01 += *biasPtr;
                    if ( ( FILTER_OFM_NUM % TILE_N ) > 8 ) blockC11 += *(biasPtr + 1);
                    #endif

                    blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                    }

                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out1[(8+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC11[i], ACTIVATION_PARAMS);
                    }
                }
                else
                {
                    #if BIAS_TERM
                    blockC01 += *biasPtr;
                    #endif

                    for (unsigned i = 0; i < FILTER_OFM_NUM % 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = ACTIVATION(blockC01[i], ACTIVATION_PARAMS);
                    }
                }
            }
        }
    }
}
