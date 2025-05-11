// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_headers/sub_group_shuffle.cl"

#define TRANSPOSE_BLOCK_8( _block )   \
        (float8)( _sub_group_shuffle( _block, 0 ), \
                  _sub_group_shuffle( _block, 1 ), \
                  _sub_group_shuffle( _block, 2 ), \
                  _sub_group_shuffle( _block, 3 ), \
                  _sub_group_shuffle( _block, 4 ), \
                  _sub_group_shuffle( _block, 5 ), \
                  _sub_group_shuffle( _block, 6 ), \
                  _sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_FP16( _block )   \
        (half8)( _sub_group_shuffle( _block, 0 ), \
                  _sub_group_shuffle( _block, 1 ), \
                  _sub_group_shuffle( _block, 2 ), \
                  _sub_group_shuffle( _block, 3 ), \
                  _sub_group_shuffle( _block, 4 ), \
                  _sub_group_shuffle( _block, 5 ), \
                  _sub_group_shuffle( _block, 6 ), \
                  _sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_COL( _block, _col )   \
        (float8)( _sub_group_shuffle( _block.s0, _col ), \
                  _sub_group_shuffle( _block.s1, _col ), \
                  _sub_group_shuffle( _block.s2, _col ), \
                  _sub_group_shuffle( _block.s3, _col ), \
                  _sub_group_shuffle( _block.s4, _col ), \
                  _sub_group_shuffle( _block.s5, _col ), \
                  _sub_group_shuffle( _block.s6, _col ), \
                  _sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_8_COL_FP16( _block, _col )   \
        (half8)( _sub_group_shuffle( _block.s0, _col ), \
                  _sub_group_shuffle( _block.s1, _col ), \
                  _sub_group_shuffle( _block.s2, _col ), \
                  _sub_group_shuffle( _block.s3, _col ), \
                  _sub_group_shuffle( _block.s4, _col ), \
                  _sub_group_shuffle( _block.s5, _col ), \
                  _sub_group_shuffle( _block.s6, _col ), \
                  _sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_16_FP16(_block)  \
        (half16)(as_half2(_sub_group_shuffle(_block, 0)),  \
                 as_half2(_sub_group_shuffle(_block, 1)),  \
                 as_half2(_sub_group_shuffle(_block, 2)),  \
                 as_half2(_sub_group_shuffle(_block, 3)),  \
                 as_half2(_sub_group_shuffle(_block, 4)),  \
                 as_half2(_sub_group_shuffle(_block, 5)),  \
                 as_half2(_sub_group_shuffle(_block, 6)),  \
                 as_half2(_sub_group_shuffle(_block, 7)));

#define TRANSPOSE_BLOCK_16_FP16_HALF_TYPE(_block)  \
        (half16)(_sub_group_shuffle(_block, 0),  \
                 _sub_group_shuffle(_block, 1),  \
                 _sub_group_shuffle(_block, 2),  \
                 _sub_group_shuffle(_block, 3),  \
                 _sub_group_shuffle(_block, 4),  \
                 _sub_group_shuffle(_block, 5),  \
                 _sub_group_shuffle(_block, 6),  \
                 _sub_group_shuffle(_block, 7),  \
                 _sub_group_shuffle(_block, 8),  \
                 _sub_group_shuffle(_block, 9),  \
                 _sub_group_shuffle(_block, 10),  \
                 _sub_group_shuffle(_block, 11),  \
                 _sub_group_shuffle(_block, 12),  \
                 _sub_group_shuffle(_block, 13),  \
                 _sub_group_shuffle(_block, 14),  \
                 _sub_group_shuffle(_block, 15));

#define TRANSPOSE_BLOCK_16(_block)  \
        (float16)(_sub_group_shuffle(_block, 0),  \
                 _sub_group_shuffle(_block, 1),  \
                 _sub_group_shuffle(_block, 2),  \
                 _sub_group_shuffle(_block, 3),  \
                 _sub_group_shuffle(_block, 4),  \
                 _sub_group_shuffle(_block, 5),  \
                 _sub_group_shuffle(_block, 6),  \
                 _sub_group_shuffle(_block, 7),  \
                 _sub_group_shuffle(_block, 8),  \
                 _sub_group_shuffle(_block, 9),  \
                 _sub_group_shuffle(_block, 10),  \
                 _sub_group_shuffle(_block, 11),  \
                 _sub_group_shuffle(_block, 12),  \
                 _sub_group_shuffle(_block, 13),  \
                 _sub_group_shuffle(_block, 14),  \
                 _sub_group_shuffle(_block, 15));

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, _sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, _sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, _sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, _sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, _sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, _sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, _sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, _sub_group_shuffle( colB, 7 ), _result.s7 );  \
}

#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += _sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += _sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += _sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += _sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += _sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += _sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += _sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += _sub_group_shuffle( _biasVal, 7 ); \
}

#define ADD_BIAS_16_FP16( _result, _biasVal) \
{ \
    _result.s01 += as_half2(_sub_group_shuffle(_biasVal, 0)); \
    _result.s23 += as_half2(_sub_group_shuffle(_biasVal, 1)); \
    _result.s45 += as_half2(_sub_group_shuffle(_biasVal, 2)); \
    _result.s67 += as_half2(_sub_group_shuffle(_biasVal, 3)); \
    _result.s89 += as_half2(_sub_group_shuffle(_biasVal, 4)); \
    _result.sab += as_half2(_sub_group_shuffle(_biasVal, 5)); \
    _result.scd += as_half2(_sub_group_shuffle(_biasVal, 6)); \
    _result.sef += as_half2(_sub_group_shuffle(_biasVal, 7)); \
}
