// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define GET_COL_8( _block, _col )                               \
        (float8)( _sub_group_shuffle( _block, _col ));

#define GET_COL_2( _block, _col )                               \
        (float2)( _sub_group_shuffle( _block, _col ));

#define GET_COL_4( _block, _col )                               \
        (float4)( _sub_group_shuffle( _block, _col ));

#define DOT4i( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s0, sub_group_broadcast( _B.s0, i), _result);	\
	_result = mad(_A.s1, sub_group_broadcast( _B.s1, i), _result);	\
	_result = mad(_A.s2, sub_group_broadcast( _B.s2, i), _result);	\
	_result = mad(_A.s3, sub_group_broadcast( _B.s3, i), _result);	\
    }

#define DOT4i_LO( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s0, sub_group_broadcast( _B.s0, i), _result);	\
	_result = mad(_A.s1, sub_group_broadcast( _B.s1, i), _result);	\
	_result = mad(_A.s2, sub_group_broadcast( _B.s2, i), _result);	\
	_result = mad(_A.s3, sub_group_broadcast( _B.s3, i), _result);	\
    }

#define DOT4i_HI( _result, _A, _B, i)					\
    {									\
	_result = mad(_A.s4, sub_group_broadcast( _B.s0, i), _result);	\
	_result = mad(_A.s5, sub_group_broadcast( _B.s1, i), _result);	\
	_result = mad(_A.s6, sub_group_broadcast( _B.s2, i), _result);	\
	_result = mad(_A.s7, sub_group_broadcast( _B.s3, i), _result);	\
    }

#define DOT8i( _result, _A, _B, i)					\
    {									\
	_result = fma(_A.s0, sub_group_broadcast( _B.s0, i), _result);	\
	_result = fma(_A.s1, sub_group_broadcast( _B.s1, i), _result);	\
	_result = fma(_A.s2, sub_group_broadcast( _B.s2, i), _result);	\
	_result = fma(_A.s3, sub_group_broadcast( _B.s3, i), _result);	\
	_result = fma(_A.s4, sub_group_broadcast( _B.s4, i), _result);	\
	_result = fma(_A.s5, sub_group_broadcast( _B.s5, i), _result);	\
	_result = fma(_A.s6, sub_group_broadcast( _B.s6, i), _result);	\
	_result = fma(_A.s7, sub_group_broadcast( _B.s7, i), _result);	\
    }

#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )                \
        {                                                               \
            const float8    acol0 = GET_COL_8( _blockA, 0 );            \
            const float8    acol1 = GET_COL_8( _blockA, 1 );            \
            const float8    acol2 = GET_COL_8( _blockA, 2 );            \
            const float8    acol3 = GET_COL_8( _blockA, 3 );            \
            const float8    acol4 = GET_COL_8( _blockA, 4 );            \
            const float8    acol5 = GET_COL_8( _blockA, 5 );            \
            const float8    acol6 = GET_COL_8( _blockA, 6 );            \
            const float8    acol7 = GET_COL_8( _blockA, 7 );            \
            _result = mad( (float8)(_blockB.s0), acol0, _result );      \
            _result = mad( (float8)(_blockB.s1), acol1, _result );      \
            _result = mad( (float8)(_blockB.s2), acol2, _result );      \
            _result = mad( (float8)(_blockB.s3), acol3, _result );      \
            _result = mad( (float8)(_blockB.s4), acol4, _result );      \
            _result = mad( (float8)(_blockB.s5), acol5, _result );      \
            _result = mad( (float8)(_blockB.s6), acol6, _result );      \
            _result = mad( (float8)(_blockB.s7), acol7, _result );      \
        }

#define MULTIPLY_BLOCKS_4x8( _result, _blockA, _blockB )                \
        {                                                               \
            const float4    acol0 = GET_COL_4( _blockA, 0 );            \
            const float4    acol1 = GET_COL_4( _blockA, 1 );            \
            const float4    acol2 = GET_COL_4( _blockA, 2 );            \
            const float4    acol3 = GET_COL_4( _blockA, 3 );            \
            const float4    acol4 = GET_COL_4( _blockA, 4 );            \
            const float4    acol5 = GET_COL_4( _blockA, 5 );            \
            const float4    acol6 = GET_COL_4( _blockA, 6 );            \
            const float4    acol7 = GET_COL_4( _blockA, 7 );            \
            _result = mad( (float4)(_blockB.s0), acol0, _result );      \
            _result = mad( (float4)(_blockB.s1), acol1, _result );      \
            _result = mad( (float4)(_blockB.s2), acol2, _result );      \
            _result = mad( (float4)(_blockB.s3), acol3, _result );      \
            _result = mad( (float4)(_blockB.s4), acol4, _result );      \
            _result = mad( (float4)(_blockB.s5), acol5, _result );      \
            _result = mad( (float4)(_blockB.s6), acol6, _result );      \
            _result = mad( (float4)(_blockB.s7), acol7, _result );      \
        }

#define MULTIPLY_BLOCKS_2x8( _result, _blockA, _blockB )                \
        {                                                               \
            const float2    acol0 = GET_COL_2( _blockA, 0 );            \
            const float2    acol1 = GET_COL_2( _blockA, 1 );            \
            const float2    acol2 = GET_COL_2( _blockA, 2 );            \
            const float2    acol3 = GET_COL_2( _blockA, 3 );            \
            const float2    acol4 = GET_COL_2( _blockA, 4 );            \
            const float2    acol5 = GET_COL_2( _blockA, 5 );            \
            const float2    acol6 = GET_COL_2( _blockA, 6 );            \
            const float2    acol7 = GET_COL_2( _blockA, 7 );            \
            _result = mad( (float2)(_blockB.s0), acol0, _result );      \
            _result = mad( (float2)(_blockB.s1), acol1, _result );      \
            _result = mad( (float2)(_blockB.s2), acol2, _result );      \
            _result = mad( (float2)(_blockB.s3), acol3, _result );      \
            _result = mad( (float2)(_blockB.s4), acol4, _result );      \
            _result = mad( (float2)(_blockB.s5), acol5, _result );      \
            _result = mad( (float2)(_blockB.s6), acol6, _result );      \
            _result = mad( (float2)(_blockB.s7), acol7, _result );      \
        }

#define MULTIPLY_BLOCKS_8x8_NO_ACCUMULATE( _result, _blockA, _blockB )  \
        {                                                               \
            const float8    acol0 = GET_COL_8( _blockA, 0 );            \
            const float8    acol1 = GET_COL_8( _blockA, 1 );            \
            const float8    acol2 = GET_COL_8( _blockA, 2 );            \
            const float8    acol3 = GET_COL_8( _blockA, 3 );            \
            const float8    acol4 = GET_COL_8( _blockA, 4 );            \
            const float8    acol5 = GET_COL_8( _blockA, 5 );            \
            const float8    acol6 = GET_COL_8( _blockA, 6 );            \
            const float8    acol7 = GET_COL_8( _blockA, 7 );            \
            _result = (float8)(_blockB.s0) * acol0;                     \
            _result = mad( (float8)(_blockB.s1), acol1, _result );      \
            _result = mad( (float8)(_blockB.s2), acol2, _result );      \
            _result = mad( (float8)(_blockB.s3), acol3, _result );      \
            _result = mad( (float8)(_blockB.s4), acol4, _result );      \
            _result = mad( (float8)(_blockB.s5), acol5, _result );      \
            _result = mad( (float8)(_blockB.s6), acol6, _result );      \
            _result = mad( (float8)(_blockB.s7), acol7, _result );      \
        }

#define MULTIPLY_BLOCKS_16x8_LO( _result, _blockA, _blockB )	\
    {								\
	const float8    acol0 = GET_COL_8( _blockA, 0 );	\
	const float8    acol1 = GET_COL_8( _blockA, 1 );	\
	const float8    acol2 = GET_COL_8( _blockA, 2 );	\
	const float8    acol3 = GET_COL_8( _blockA, 3 );	\
	const float8    acol4 = GET_COL_8( _blockA, 4 );	\
	const float8    acol5 = GET_COL_8( _blockA, 5 );	\
	const float8    acol6 = GET_COL_8( _blockA, 6 );	\
	const float8    acol7 = GET_COL_8( _blockA, 7 );	\
	_result = mad( (float8)(_blockB.s0), acol0, _result );	\
	_result = mad( (float8)(_blockB.s1), acol1, _result );	\
	_result = mad( (float8)(_blockB.s2), acol2, _result );	\
	_result = mad( (float8)(_blockB.s3), acol3, _result );	\
	_result = mad( (float8)(_blockB.s4), acol4, _result );	\
	_result = mad( (float8)(_blockB.s5), acol5, _result );	\
	_result = mad( (float8)(_blockB.s6), acol6, _result );	\
	_result = mad( (float8)(_blockB.s7), acol7, _result );	\
    }

#define MULTIPLY_BLOCKS_16x8_HI( _result, _blockA, _blockB )	\
    {								\
	const float8    acol0 = GET_COL_8( _blockA, 8 );	\
	const float8    acol1 = GET_COL_8( _blockA, 9 );	\
	const float8    acol2 = GET_COL_8( _blockA, 10 );	\
	const float8    acol3 = GET_COL_8( _blockA, 11 );	\
	const float8    acol4 = GET_COL_8( _blockA, 12 );	\
	const float8    acol5 = GET_COL_8( _blockA, 13 );	\
	const float8    acol6 = GET_COL_8( _blockA, 14 );	\
	const float8    acol7 = GET_COL_8( _blockA, 15 );	\
	_result = mad( (float8)(_blockB.s0), acol0, _result );	\
	_result = mad( (float8)(_blockB.s1), acol1, _result );	\
	_result = mad( (float8)(_blockB.s2), acol2, _result );	\
	_result = mad( (float8)(_blockB.s3), acol3, _result );	\
	_result = mad( (float8)(_blockB.s4), acol4, _result );	\
	_result = mad( (float8)(_blockB.s5), acol5, _result );	\
	_result = mad( (float8)(_blockB.s6), acol6, _result );	\
	_result = mad( (float8)(_blockB.s7), acol7, _result );	\
    }


#define WRITE_BLOCK_2(ptr_,     block0_, block1_, row_)                 \
    if (row_ < max_row) {                                               \
        const float2 vec = (float2)(block0_.s ## row_,                  \
                                    block1_.s ## row_);                 \
        _sub_group_block_write2((__global uint*)&ptr_[N*row_], as_uint2(vec)); \
    }

#define WRITE_BLOCK_4(ptr_, block0_, block1_, block2_, block3_, row_)   \
    if (row_ < max_row) {                                               \
        const float4 vec = (float4)(block0_.s ## row_,                  \
                                    block1_.s ## row_,                  \
                                    block2_.s ## row_,                  \
                                    block3_.s ## row_);                 \
        _sub_group_block_write4((__global uint*)&ptr_[N*row_], as_uint4(vec)); \
    }
