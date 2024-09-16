// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_headers/common.cl"
#include "batch_headers/sub_group_block_read.cl"
#include "batch_headers/sub_group_block_write.cl"

#ifndef UNIT_TYPE
#error Unit type not defined.
#endif

#if UNIT_TYPE_SIZE == 2
// 2 byte data type case (ex. half)
#define UNIT_BLOCK_RW_TYPE ushort
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read_us
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write_us
#elif UNIT_TYPE_SIZE == 4
// 4 byte data type case (ex. float)
#define UNIT_BLOCK_RW_TYPE uint
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write
#else
#error Unsupported unit type for block read/write.
#endif

#define UNIT_TYPE2 MAKE_VECTOR_TYPE(UNIT_TYPE, 2)
#define UNIT_TYPE4 MAKE_VECTOR_TYPE(UNIT_TYPE, 4)
#define UNIT_TYPE8 MAKE_VECTOR_TYPE(UNIT_TYPE, 8)

#define UNIT_BLOCK_RW_TYPE2 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 2)
#define UNIT_BLOCK_RW_TYPE4 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 4)
#define UNIT_BLOCK_RW_TYPE8 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 8)


#define UNIT_BLOCK_READ_FUNC2 CAT(UNIT_BLOCK_READ_FUNC, 2)
#define UNIT_BLOCK_READ_FUNC4 CAT(UNIT_BLOCK_READ_FUNC, 4)
#define UNIT_BLOCK_READ_FUNC8 CAT(UNIT_BLOCK_READ_FUNC, 8)

#define UNIT_BLOCK_WRITE_FUNC2 CAT(UNIT_BLOCK_WRITE_FUNC, 2)
#define UNIT_BLOCK_WRITE_FUNC4 CAT(UNIT_BLOCK_WRITE_FUNC, 4)
#define UNIT_BLOCK_WRITE_FUNC8 CAT(UNIT_BLOCK_WRITE_FUNC, 8)

#define UNIT_BLOCK_READ(ptr, offset)  AS_TYPE(UNIT_TYPE,  UNIT_BLOCK_READ_FUNC( (const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ2(ptr, offset) AS_TYPE(UNIT_TYPE2, UNIT_BLOCK_READ_FUNC2((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ4(ptr, offset) AS_TYPE(UNIT_TYPE4, UNIT_BLOCK_READ_FUNC4((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ8(ptr, offset) AS_TYPE(UNIT_TYPE8, UNIT_BLOCK_READ_FUNC8((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))

#define UNIT_BLOCK_WRITE(ptr, offset, val)  UNIT_BLOCK_WRITE_FUNC( (__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE,  val))
#define UNIT_BLOCK_WRITE2(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC2((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE2, val))
#define UNIT_BLOCK_WRITE4(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC4((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE4, val))
#define UNIT_BLOCK_WRITE8(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC8((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE8, val))
