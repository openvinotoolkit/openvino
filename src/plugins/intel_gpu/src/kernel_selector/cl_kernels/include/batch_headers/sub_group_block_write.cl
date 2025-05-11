// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.cl"

// ====================================================================================================================
// BLOCK_WRITEN(type, vector_size, ptr, offset, val)
//    - evaluates to intel_sub_group_block_write operation for specified "type" and "vector size", writing
//      "vector_size"-element vector "val" to memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read/write functions please,
//  refer to cl_intel_subgroups extension documentation.
//
// BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)
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
//   DT_<tensor>_BLOCK_WRITE<n>(ptr, offset, offset)
// Where:
//    <tensor> is usually OUTPUT,
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will be a scalar
//
// ====================================================================================================================

#define BLOCK_WRITE_TYPE_size1 uchar
#define BLOCK_WRITE_TYPE_size2 ushort
#define BLOCK_WRITE_TYPE_size4 uint
#define BLOCK_WRITE_TYPE_size8 ulong
#define BLOCK_WRITE_TYPE(type_size) CAT(BLOCK_WRITE_TYPE_size, type_size)

#define BLOCK_WRITE_FUNC_size1       _sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_size2       _sub_group_block_write_us
#define BLOCK_WRITE_FUNC_size4       _sub_group_block_write
#define BLOCK_WRITE_FUNC_size8       _sub_group_block_write_ul
#define BLOCK_WRITE_FUNC(type_size)  CAT(BLOCK_WRITE_FUNC_size, type_size)

#define BLOCK_WRITEN_FUNC_SIZE_DEF(type_size, vector_size)  MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vector_size)
#define BLOCK_WRITEN_FUNC_size1(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_WRITEN_FUNC_size2(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_WRITEN_FUNC_size4(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_WRITEN_FUNC_size8(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_WRITEN_FUNC(type_size, vector_size)           CAT(BLOCK_WRITEN_FUNC_size, type_size)(vector_size)

#define BLOCK_WRITEN_RAW(type_size, vector_size, addr_space, ptr, offset, val)                                  \
    BLOCK_WRITEN_FUNC(type_size, vector_size)(                                                                  \
        (addr_space BLOCK_WRITE_TYPE(type_size)*)(ptr) + (offset),                                              \
        AS_TYPE(MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vector_size), val))

#define BLOCK_WRITEN(type, vector_size, ptr, offset, val)                                                       \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset, val)

#define BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)                                                   \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset, val)

#define DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)     BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE2(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 2, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE4(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 4, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE8(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 8, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE16(ptr, offset, val)   BLOCK_WRITEN(OUTPUT_TYPE, 16, ptr, offset, val)

#define BLOCK_WRITE_IMPL_1 out_ptr[idx] = v;
#define BLOCK_WRITE_IMPL_2                                    \
        out_ptr[idx] = v.s0; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s1; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_4                                    \
        BLOCK_WRITE_IMPL_2                                    \
        out_ptr[idx] = v.s2; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s3; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_8                                    \
        BLOCK_WRITE_IMPL_4                                    \
        out_ptr[idx] = v.s4; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s5; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s6; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s7; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_16                                   \
        BLOCK_WRITE_IMPL_8                                    \
        out_ptr[idx] = v.s8; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s9; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sa; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sb; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sc; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sd; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.se; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sf; idx += get_max_sub_group_size();

#define BLOCK_WRITE_IMPL(vec_size) CAT(BLOCK_WRITE_IMPL_, vec_size)
#define BLOCK_WRITE_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_WRITE_EMULATION(type_size, vec_size) \
    inline void BLOCK_WRITE_FUNC_NAME(type_size, vec_size)(__global BLOCK_WRITE_TYPE(type_size)* out_ptr, \
                                                           MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vec_size) v) { \
    uint idx = get_sub_group_local_id(); \
    BLOCK_WRITE_IMPL(vec_size) \
}

#if defined(cl_intel_subgroups)
    #define _sub_group_block_write(ptr, v) intel_sub_group_block_write(ptr, v)
    #define _sub_group_block_write2(ptr, v) intel_sub_group_block_write2(ptr, v)
    #define _sub_group_block_write4(ptr, v) intel_sub_group_block_write4(ptr, v)
    #define _sub_group_block_write8(ptr, v) intel_sub_group_block_write8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(4, 1)
    DECLARE_BLOCK_WRITE_EMULATION(4, 2)
    DECLARE_BLOCK_WRITE_EMULATION(4, 4)
    DECLARE_BLOCK_WRITE_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroups_short)
    #define _sub_group_block_write_us(ptr, v) intel_sub_group_block_write_us(ptr, v)
    #define _sub_group_block_write_us2(ptr, v) intel_sub_group_block_write_us2(ptr, v)
    #define _sub_group_block_write_us4(ptr, v) intel_sub_group_block_write_us4(ptr, v)
    #define _sub_group_block_write_us8(ptr, v) intel_sub_group_block_write_us8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(2, 1)
    DECLARE_BLOCK_WRITE_EMULATION(2, 2)
    DECLARE_BLOCK_WRITE_EMULATION(2, 4)
    DECLARE_BLOCK_WRITE_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
    #define _sub_group_block_write_uc(ptr, v) intel_sub_group_block_write_uc(ptr, v)
    #define _sub_group_block_write_uc2(ptr, v) intel_sub_group_block_write_uc2(ptr, v)
    #define _sub_group_block_write_uc4(ptr, v) intel_sub_group_block_write_uc4(ptr, v)
    #define _sub_group_block_write_uc8(ptr, v) intel_sub_group_block_write_uc8(ptr, v)
    #define _sub_group_block_write_uc16(ptr, v) intel_sub_group_block_write_uc16(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(1, 1)
    DECLARE_BLOCK_WRITE_EMULATION(1, 2)
    DECLARE_BLOCK_WRITE_EMULATION(1, 4)
    DECLARE_BLOCK_WRITE_EMULATION(1, 8)
    DECLARE_BLOCK_WRITE_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroups_long)
    #define _sub_group_block_write_ul(ptr, v)  intel_sub_group_block_write_ul(ptr, v)
    #define _sub_group_block_write_ul2(ptr, v) intel_sub_group_block_write_ul2(ptr, v)
    #define _sub_group_block_write_ul4(ptr, v) intel_sub_group_block_write_ul4(ptr, v)
    #define _sub_group_block_write_ul8(ptr, v) intel_sub_group_block_write_ul8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(8, 1)
    DECLARE_BLOCK_WRITE_EMULATION(8, 2)
    DECLARE_BLOCK_WRITE_EMULATION(8, 4)
    DECLARE_BLOCK_WRITE_EMULATION(8, 8)
#endif
