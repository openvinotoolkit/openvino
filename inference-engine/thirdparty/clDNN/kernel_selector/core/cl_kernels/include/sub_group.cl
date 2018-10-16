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

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_intel_subgroups_short)
#pragma OPENCL EXTENSION  cl_intel_subgroups_short : enable
#endif

#define TRANSPOSE_BLOCK_8( _block )   \
        (float8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_FP16( _block )   \
        (half8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_COL( _block, _col )   \
        (float8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_8_COL_FP16( _block, _col )   \
        (half8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_16_FP16(_block)  \
        (half16)(as_half2(intel_sub_group_shuffle(_block, 0)),  \
                 as_half2(intel_sub_group_shuffle(_block, 1)),  \
                 as_half2(intel_sub_group_shuffle(_block, 2)),  \
                 as_half2(intel_sub_group_shuffle(_block, 3)),  \
                 as_half2(intel_sub_group_shuffle(_block, 4)),  \
                 as_half2(intel_sub_group_shuffle(_block, 5)),  \
                 as_half2(intel_sub_group_shuffle(_block, 6)),  \
                 as_half2(intel_sub_group_shuffle(_block, 7)));

#define TRANSPOSE_BLOCK_16_FP16_HALF_TYPE(_block)  \
        (half16)(intel_sub_group_shuffle(_block, 0),  \
                 intel_sub_group_shuffle(_block, 1),  \
                 intel_sub_group_shuffle(_block, 2),  \
                 intel_sub_group_shuffle(_block, 3),  \
                 intel_sub_group_shuffle(_block, 4),  \
                 intel_sub_group_shuffle(_block, 5),  \
                 intel_sub_group_shuffle(_block, 6),  \
                 intel_sub_group_shuffle(_block, 7),  \
                 intel_sub_group_shuffle(_block, 8),  \
                 intel_sub_group_shuffle(_block, 9),  \
                 intel_sub_group_shuffle(_block, 10),  \
                 intel_sub_group_shuffle(_block, 11),  \
                 intel_sub_group_shuffle(_block, 12),  \
                 intel_sub_group_shuffle(_block, 13),  \
                 intel_sub_group_shuffle(_block, 14),  \
                 intel_sub_group_shuffle(_block, 15));

#define TRANSPOSE_BLOCK_16(_block)  \
        (float16)(intel_sub_group_shuffle(_block, 0),  \
                 intel_sub_group_shuffle(_block, 1),  \
                 intel_sub_group_shuffle(_block, 2),  \
                 intel_sub_group_shuffle(_block, 3),  \
                 intel_sub_group_shuffle(_block, 4),  \
                 intel_sub_group_shuffle(_block, 5),  \
                 intel_sub_group_shuffle(_block, 6),  \
                 intel_sub_group_shuffle(_block, 7),  \
                 intel_sub_group_shuffle(_block, 8),  \
                 intel_sub_group_shuffle(_block, 9),  \
                 intel_sub_group_shuffle(_block, 10),  \
                 intel_sub_group_shuffle(_block, 11),  \
                 intel_sub_group_shuffle(_block, 12),  \
                 intel_sub_group_shuffle(_block, 13),  \
                 intel_sub_group_shuffle(_block, 14),  \
                 intel_sub_group_shuffle(_block, 15));

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
}

#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
}

#define ADD_BIAS_16_FP16( _result, _biasVal) \
{ \
    _result.s01 += as_half2(intel_sub_group_shuffle(_biasVal, 0)); \
    _result.s23 += as_half2(intel_sub_group_shuffle(_biasVal, 1)); \
    _result.s45 += as_half2(intel_sub_group_shuffle(_biasVal, 2)); \
    _result.s67 += as_half2(intel_sub_group_shuffle(_biasVal, 3)); \
    _result.s89 += as_half2(intel_sub_group_shuffle(_biasVal, 4)); \
    _result.sab += as_half2(intel_sub_group_shuffle(_biasVal, 5)); \
    _result.scd += as_half2(intel_sub_group_shuffle(_biasVal, 6)); \
    _result.sef += as_half2(intel_sub_group_shuffle(_biasVal, 7)); \
}
