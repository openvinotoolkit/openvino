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

#include "mmad.cl"

// TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
#if !defined(ACCUMULATOR_TYPE)
    #define ACCUMULATOR_TYPE float
    #define TO_ACCUMULATOR_TYPE(v) (float)(v)
    #define ACCUMULATOR_TYPE_ZERO 0.0f
#endif

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE(type, val) CAT(as_, type)(val)

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
#define TYPE_SIZE_uchar  1
#define TYPE_SIZE_char   1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short  2
#define TYPE_SIZE_half   2
#define TYPE_SIZE_int    4
#define TYPE_SIZE_uint   4
#define TYPE_SIZE_float  4
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

// ====================================================================================================================
// BLOCK_READN(type, vector_size, ptr, offset)
//    - evaluates to intel_sub_group_block_read operation for specified "type" and "vector size", reading
//      "vector_size" elements from memory starting at "ptr" + "offset"
// BLOCK_WRITEN(type, vector_size, ptr, offset, val)
//    - evaluates to intel_sub_group_block_write operation for specified "type" and "vector size", writing
//      "vector_size"-element vector "val" to memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read/write functions please,
//  refer to cl_intel_subgroups extension documentation.
//
// type        [PP] - Must evaluate to non-vectorized type, ex. float, half, char, etc..
// vector_size [PP] - Number of elements to read/write, ex 2 for intel_sub_group_block_read2.
// ptr              - Pointer to global memory where to read from/write to.
// offset           - Additional offset added to ptr in "type" elements, equivalent to passing ((ptr) + (offset)) as "ptr".
// val              - For write function vector of "vector_size" of "type" elements (or scalar) to write.
//
// ====================================================================================================================
// Pre-defined commonly used definitions:
//   DT_<tensor>_BLOCK_READ<n>(ptr, offset)
//   DT_<tensor>_BLOCK_WRITE<n>(ptr, offset, offset)
// Where:
//    <tensor> is one of: INPUT - referencing type jitted as INPUT0,
//                        OUTPUT,
//                        BIAS,
//                        FILTER
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will be a scalar
// 
// ====================================================================================================================
#define BLOCK_RW_TYPE_size1 uchar
#define BLOCK_RW_TYPE_size2 ushort
#define BLOCK_RW_TYPE_size4 uint
#define BLOCK_RW_TYPE(type_size) CAT(BLOCK_RW_TYPE_size, type_size)

#define BLOCK_READ_FUNC_size2       intel_sub_group_block_read_us
#define BLOCK_READ_FUNC_size4       intel_sub_group_block_read
#define BLOCK_READ_FUNC(type_size)  CAT(BLOCK_READ_FUNC_size, type_size)

#define BLOCK_WRITE_FUNC_size2       intel_sub_group_block_write_us
#define BLOCK_WRITE_FUNC_size4       intel_sub_group_block_write
#define BLOCK_WRITE_FUNC(type_size)  CAT(BLOCK_WRITE_FUNC_size, type_size)

#define BLOCK_READN_FUNC_size1(vector_size)                 CAT(BLOCK_READ_UC_, vector_size)
#define BLOCK_READN_FUNC_SIZE_DEF(type_size, vector_size)   MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vector_size)
#define BLOCK_READN_FUNC_size2(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_size4(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC(type_size, vector_size)            CAT(BLOCK_READN_FUNC_size, type_size)(vector_size)

#define BLOCK_WRITEN_FUNC_size1(vector_size)                CAT(BLOCK_WRITE_UC_, vector_size)
#define BLOCK_WRITEN_FUNC_SIZE_DEF(type_size, vector_size)  MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vector_size)
#define BLOCK_WRITEN_FUNC_size2(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_WRITEN_FUNC_size4(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_WRITEN_FUNC(type_size, vector_size)           CAT(BLOCK_WRITEN_FUNC_size, type_size)(vector_size)

#define BLOCK_READN_RAW(type_size, vector_size, ptr, offset)                                                    \
    BLOCK_READN_FUNC(type_size, vector_size)((const __global BLOCK_RW_TYPE(type_size)*)(ptr) + (offset))
#define BLOCK_WRITEN_RAW(type_size, vector_size, ptr, offset, val)                                              \
    BLOCK_WRITEN_FUNC(type_size, vector_size)(                                                                  \
        (__global BLOCK_RW_TYPE(type_size)*)(ptr) + (offset),                                                   \
        AS_TYPE(MAKE_VECTOR_TYPE(BLOCK_RW_TYPE(type_size), vector_size), val))

#define BLOCK_READN(type, vector_size, ptr, offset)                                                             \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, ptr, offset))
#define BLOCK_WRITEN(type, vector_size, ptr, offset, val)                                                       \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, ptr, offset, val)

#define DT_INPUT_BLOCK_READ(ptr, offset)            BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset)          BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)

#define DT_INPUT_BLOCK_WRITE(ptr, offset, val)      BLOCK_WRITEN(INPUT0_TYPE, 1, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE2(ptr, offset, val)     BLOCK_WRITEN(INPUT0_TYPE, 2, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE4(ptr, offset, val)     BLOCK_WRITEN(INPUT0_TYPE, 4, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE8(ptr, offset, val)     BLOCK_WRITEN(INPUT0_TYPE, 8, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE16(ptr, offset, val)    BLOCK_WRITEN(INPUT0_TYPE, 16, ptr, offset, val)

#define DT_OUTPUT_BLOCK_READ(ptr, offset)           BLOCK_READN(OUTPUT_TYPE, 1, ptr, offset)
#define DT_OUTPUT_BLOCK_READ2(ptr, offset)          BLOCK_READN(OUTPUT_TYPE, 2, ptr, offset)
#define DT_OUTPUT_BLOCK_READ4(ptr, offset)          BLOCK_READN(OUTPUT_TYPE, 4, ptr, offset)
#define DT_OUTPUT_BLOCK_READ8(ptr, offset)          BLOCK_READN(OUTPUT_TYPE, 8, ptr, offset)
#define DT_OUTPUT_BLOCK_READ16(ptr, offset)         BLOCK_READN(OUTPUT_TYPE, 16, ptr, offset)

#define DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)     BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE2(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 2, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE4(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 4, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE8(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 8, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE16(ptr, offset, val)   BLOCK_WRITEN(OUTPUT_TYPE, 16, ptr, offset, val)

#define DT_BIAS_BLOCK_READ(ptr, offset)             BLOCK_READN(BIAS_TYPE, 1, ptr, offset)
#define DT_BIAS_BLOCK_READ2(ptr, offset)            BLOCK_READN(BIAS_TYPE, 2, ptr, offset)
#define DT_BIAS_BLOCK_READ4(ptr, offset)            BLOCK_READN(BIAS_TYPE, 4, ptr, offset)
#define DT_BIAS_BLOCK_READ8(ptr, offset)            BLOCK_READN(BIAS_TYPE, 8, ptr, offset)
#define DT_BIAS_BLOCK_READ16(ptr, offset)           BLOCK_READN(BIAS_TYPE, 16, ptr, offset)

#define DT_BIAS_BLOCK_WRITE(ptr, offset, val)       BLOCK_WRITEN(BIAS_TYPE, 1, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE2(ptr, offset, val)      BLOCK_WRITEN(BIAS_TYPE, 2, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE4(ptr, offset, val)      BLOCK_WRITEN(BIAS_TYPE, 4, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE8(ptr, offset, val)      BLOCK_WRITEN(BIAS_TYPE, 8, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE16(ptr, offset, val)     BLOCK_WRITEN(BIAS_TYPE, 16, ptr, offset, val)

#define DT_FILTER_BLOCK_READ(ptr, offset)           BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset)          BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset)          BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset)          BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset)         BLOCK_READN(FILTER_TYPE, 16, ptr, offset)

#define DT_FILTER_BLOCK_WRITE(ptr, offset, val)     BLOCK_WRITEN(FILTER_TYPE, 1, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE2(ptr, offset, val)    BLOCK_WRITEN(FILTER_TYPE, 2, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE4(ptr, offset, val)    BLOCK_WRITEN(FILTER_TYPE, 4, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE8(ptr, offset, val)    BLOCK_WRITEN(FILTER_TYPE, 8, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE16(ptr, offset, val)   BLOCK_WRITEN(FILTER_TYPE, 16, ptr, offset, val)
// ====================================================================================================================
