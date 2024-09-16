// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.cl"

// ====================================================================================================================
// BLOCK_READN(type, vector_size, ptr, offset)
//    - evaluates to intel_sub_group_block_read operation for specified "type" and "vector size", reading
//      "vector_size" elements from memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read functions please,
//  refer to cl_intel_subgroups extension documentation.
//
// BLOCK_READN_SLM(type, vector_size, ptr, offset)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local address space.
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
// Where:
//    <tensor> is one of: INPUT - referencing type jitted as INPUT0,
//                        BIAS,
//                        FILTER
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will be a scalar
//
// ====================================================================================================================

#define BLOCK_READ_TYPE_size1 uchar
#define BLOCK_READ_TYPE_size2 ushort
#define BLOCK_READ_TYPE_size4 uint
#define BLOCK_READ_TYPE_size8 ulong
#define BLOCK_READ_TYPE(type_size) CAT(BLOCK_READ_TYPE_size, type_size)

#define BLOCK_READ_FUNC_size1       _sub_group_block_read_uc
#define BLOCK_READ_FUNC_size2       _sub_group_block_read_us
#define BLOCK_READ_FUNC_size4       _sub_group_block_read
#define BLOCK_READ_FUNC_size8       _sub_group_block_read_ul
#define BLOCK_READ_FUNC(type_size)  CAT(BLOCK_READ_FUNC_size, type_size)

#define BLOCK_READN_FUNC_SIZE_DEF(type_size, vector_size)   MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vector_size)
#define BLOCK_READN_FUNC_size1(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_READN_FUNC_size2(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_size4(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC_size8(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_READN_FUNC(type_size, vector_size)            CAT(BLOCK_READN_FUNC_size, type_size)(vector_size)

#define BLOCK_READN_RAW(type_size, vector_size, addr_space, ptr, offset)                                        \
    BLOCK_READN_FUNC(type_size, vector_size)((const addr_space BLOCK_READ_TYPE(type_size)*)(ptr) + (offset))

#define BLOCK_READN(type, vector_size, ptr, offset)                                                             \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset))

#define BLOCK_READN_SLM(type, vector_size, ptr, offset)                                                         \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset))

#define DT_INPUT_BLOCK_READ(ptr, offset)            BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset)          BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)

#define DT_BIAS_BLOCK_READ(ptr, offset)             BLOCK_READN(BIAS_TYPE, 1, ptr, offset)
#define DT_BIAS_BLOCK_READ2(ptr, offset)            BLOCK_READN(BIAS_TYPE, 2, ptr, offset)
#define DT_BIAS_BLOCK_READ4(ptr, offset)            BLOCK_READN(BIAS_TYPE, 4, ptr, offset)
#define DT_BIAS_BLOCK_READ8(ptr, offset)            BLOCK_READN(BIAS_TYPE, 8, ptr, offset)
#define DT_BIAS_BLOCK_READ16(ptr, offset)           BLOCK_READN(BIAS_TYPE, 16, ptr, offset)

#define DT_FILTER_BLOCK_READ(ptr, offset)           BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset)          BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset)          BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset)          BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset)         BLOCK_READN(FILTER_TYPE, 16, ptr, offset)


#define BLOCK_READ_IMPL_1 ret = ptr[idx];

#define BLOCK_READ_IMPL_2                                   \
        ret.s0 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s1 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_4                                   \
        BLOCK_READ_IMPL_2                                   \
        ret.s2 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s3 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_8                                   \
        BLOCK_READ_IMPL_4                                   \
        ret.s4 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s5 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s6 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s7 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_16                                  \
        BLOCK_READ_IMPL_8                                   \
        ret.s8 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s9 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sa = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sb = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sc = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sd = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.se = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sf = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL(vec_size) CAT(BLOCK_READ_IMPL_, vec_size)
#define BLOCK_READ_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_READ_EMULATION(type_size, vec_size) \
    inline MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) BLOCK_READ_FUNC_NAME(type_size, vec_size)(const __global BLOCK_READ_TYPE(type_size)* ptr) { \
    uint idx = get_sub_group_local_id(); \
    MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) ret; \
    BLOCK_READ_IMPL(vec_size) \
    return ret; \
}

#if defined(cl_intel_subgroups)
    #define _sub_group_block_read(ptr) intel_sub_group_block_read(ptr)
    #define _sub_group_block_read2(ptr) intel_sub_group_block_read2(ptr)
    #define _sub_group_block_read4(ptr) intel_sub_group_block_read4(ptr)
    #define _sub_group_block_read8(ptr) intel_sub_group_block_read8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(4, 1)
    DECLARE_BLOCK_READ_EMULATION(4, 2)
    DECLARE_BLOCK_READ_EMULATION(4, 4)
    DECLARE_BLOCK_READ_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroups_short)
    #define _sub_group_block_read_us(ptr) intel_sub_group_block_read_us(ptr)
    #define _sub_group_block_read_us2(ptr) intel_sub_group_block_read_us2(ptr)
    #define _sub_group_block_read_us4(ptr) intel_sub_group_block_read_us4(ptr)
    #define _sub_group_block_read_us8(ptr) intel_sub_group_block_read_us8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(2, 1)
    DECLARE_BLOCK_READ_EMULATION(2, 2)
    DECLARE_BLOCK_READ_EMULATION(2, 4)
    DECLARE_BLOCK_READ_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
    #define _sub_group_block_read_uc(ptr) intel_sub_group_block_read_uc(ptr)
    #define _sub_group_block_read_uc2(ptr) intel_sub_group_block_read_uc2(ptr)
    #define _sub_group_block_read_uc4(ptr) intel_sub_group_block_read_uc4(ptr)
    #define _sub_group_block_read_uc8(ptr) intel_sub_group_block_read_uc8(ptr)
    #define _sub_group_block_read_uc16(ptr) intel_sub_group_block_read_uc16(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(1, 1)
    DECLARE_BLOCK_READ_EMULATION(1, 2)
    DECLARE_BLOCK_READ_EMULATION(1, 4)
    DECLARE_BLOCK_READ_EMULATION(1, 8)
    DECLARE_BLOCK_READ_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroups_long)
    #define _sub_group_block_read_ul(ptr)  intel_sub_group_block_read_ul(ptr)
    #define _sub_group_block_read_ul2(ptr) intel_sub_group_block_read_ul2(ptr)
    #define _sub_group_block_read_ul4(ptr) intel_sub_group_block_read_ul4(ptr)
    #define _sub_group_block_read_ul8(ptr) intel_sub_group_block_read_ul8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(8, 1)
    DECLARE_BLOCK_READ_EMULATION(8, 2)
    DECLARE_BLOCK_READ_EMULATION(8, 4)
    DECLARE_BLOCK_READ_EMULATION(8, 8)
#endif
