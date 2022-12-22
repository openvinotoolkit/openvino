// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


inline void sub_group_block_write_uchar16(__global uchar* outPtr, uchar16 v) {
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc16(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
    outPtr[idx] = v.s2; idx += get_max_sub_group_size();
    outPtr[idx] = v.s3; idx += get_max_sub_group_size();
    outPtr[idx] = v.s4; idx += get_max_sub_group_size();
    outPtr[idx] = v.s5; idx += get_max_sub_group_size();
    outPtr[idx] = v.s6; idx += get_max_sub_group_size();
    outPtr[idx] = v.s7; idx += get_max_sub_group_size();
    outPtr[idx] = v.s8; idx += get_max_sub_group_size();
    outPtr[idx] = v.s9; idx += get_max_sub_group_size();
    outPtr[idx] = v.sa; idx += get_max_sub_group_size();
    outPtr[idx] = v.sb; idx += get_max_sub_group_size();
    outPtr[idx] = v.sc; idx += get_max_sub_group_size();
    outPtr[idx] = v.sd; idx += get_max_sub_group_size();
    outPtr[idx] = v.se; idx += get_max_sub_group_size();
    outPtr[idx] = v.sf; idx += get_max_sub_group_size();
#endif
}

inline uchar16 sub_group_block_read_uchar16(const __global uchar* ptr) __attribute__((overloadable)) {
#ifdef cl_intel_subgroups_char
    // WA for compiler support
    // return intel_sub_group_block_read_uc16(ptr);
    return (uchar16)(intel_sub_group_block_read_uc8(ptr), intel_sub_group_block_read_uc8(ptr + 8 * get_max_sub_group_size()));
#else
    uint idx = get_sub_group_local_id();

    uchar16 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s8 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s9 = ptr[idx]; idx += get_max_sub_group_size();
    ret.sa = ptr[idx]; idx += get_max_sub_group_size();
    ret.sb = ptr[idx]; idx += get_max_sub_group_size();
    ret.sc = ptr[idx]; idx += get_max_sub_group_size();
    ret.sd = ptr[idx]; idx += get_max_sub_group_size();
    ret.se = ptr[idx]; idx += get_max_sub_group_size();
    ret.sf = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar16 sub_group_block_read_uchar16(const __local uchar* ptr) __attribute__((overloadable)) {
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    // WA for compiler support
    // return intel_sub_group_block_read_uc16(ptr);
    return (uchar16)(intel_sub_group_block_read_uc8(ptr), intel_sub_group_block_read_uc8(ptr + 8 * get_max_sub_group_size()));
#else
    uint idx = get_sub_group_local_id();

    uchar16 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s8 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s9 = ptr[idx]; idx += get_max_sub_group_size();
    ret.sa = ptr[idx]; idx += get_max_sub_group_size();
    ret.sb = ptr[idx]; idx += get_max_sub_group_size();
    ret.sc = ptr[idx]; idx += get_max_sub_group_size();
    ret.sd = ptr[idx]; idx += get_max_sub_group_size();
    ret.se = ptr[idx]; idx += get_max_sub_group_size();
    ret.sf = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void sub_group_block_write_uchar8(__global uchar* outPtr, uchar8 v)
{
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc8(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
    outPtr[idx] = v.s2; idx += get_max_sub_group_size();
    outPtr[idx] = v.s3; idx += get_max_sub_group_size();
    outPtr[idx] = v.s4; idx += get_max_sub_group_size();
    outPtr[idx] = v.s5; idx += get_max_sub_group_size();
    outPtr[idx] = v.s6; idx += get_max_sub_group_size();
    outPtr[idx] = v.s7; idx += get_max_sub_group_size();
#endif
}

inline uchar8 sub_group_block_read_uchar8(const __global uchar* ptr) __attribute__((overloadable)) {
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc8(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar8 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar8 sub_group_block_read_uchar8(const __local uchar* ptr) __attribute__((overloadable)) {
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc8(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar8 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void sub_group_block_write_uchar4(__global uchar* outPtr, uchar4 v) {
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc4(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
    outPtr[idx] = v.s2; idx += get_max_sub_group_size();
    outPtr[idx] = v.s3; idx += get_max_sub_group_size();
#endif
}

inline uchar4 sub_group_block_read_uchar4(const __global uchar* ptr) __attribute__((overloadable)) {
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc4(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar4 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar4 sub_group_block_read_uchar4(const __local uchar* ptr) __attribute__((overloadable)) {
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc4(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar4 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void sub_group_block_write_uchar2(__global uchar* outPtr, uchar2 v) {
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc2(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
#endif
}

inline uchar2 sub_group_block_read_uchar2(const __global uchar* ptr) __attribute__((overloadable)) {
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc2(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar2 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar2 sub_group_block_read_uchar2(const __local uchar* ptr) __attribute__((overloadable)) {
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc2(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar2 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void sub_group_block_write_uchar(__global uchar* outPtr, uchar v) {
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v;
#endif
}

inline uchar sub_group_block_read_uchar(const __global uchar* ptr) __attribute__((overloadable)) {
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar ret;

    ret = ptr[idx];

    return ret;
#endif
}

inline uchar sub_group_block_read_uchar(const __local uchar* ptr) __attribute__((overloadable)) {
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar ret;

    ret = ptr[idx];

    return ret;
#endif
}

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
// BLOCK_READN_SLM(type, vector_size, ptr, offset)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local address space.
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

#define BLOCK_READ_UC_1(ptr)  sub_group_block_read_uchar(ptr)
#define BLOCK_READ_UC_2(ptr)  sub_group_block_read_uchar2(ptr)
#define BLOCK_READ_UC_4(ptr)  sub_group_block_read_uchar4(ptr)
#define BLOCK_READ_UC_8(ptr)  sub_group_block_read_uchar8(ptr)
#define BLOCK_READ_UC_16(ptr) sub_group_block_read_uchar16(ptr)

#define BLOCK_WRITE_UC_1(ptr, val)  sub_group_block_write_uchar(ptr, val)
#define BLOCK_WRITE_UC_2(ptr, val)  sub_group_block_write_uchar2(ptr, val)
#define BLOCK_WRITE_UC_4(ptr, val)  sub_group_block_write_uchar4(ptr, val)
#define BLOCK_WRITE_UC_8(ptr, val)  sub_group_block_write_uchar8(ptr, val)
#define BLOCK_WRITE_UC_16(ptr, val) sub_group_block_write_uchar16(ptr, val)

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

#define BLOCK_READN_RAW(type_size, vector_size, addr_space, ptr, offset)                                        \
    BLOCK_READN_FUNC(type_size, vector_size)((const addr_space BLOCK_RW_TYPE(type_size)*)(ptr) + (offset))
#define BLOCK_WRITEN_RAW(type_size, vector_size, addr_space, ptr, offset, val)                                  \
    BLOCK_WRITEN_FUNC(type_size, vector_size)(                                                                  \
        (addr_space BLOCK_RW_TYPE(type_size)*)(ptr) + (offset),                                                 \
        AS_TYPE(MAKE_VECTOR_TYPE(BLOCK_RW_TYPE(type_size), vector_size), val))

#define BLOCK_READN(type, vector_size, ptr, offset)                                                             \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset))
#define BLOCK_WRITEN(type, vector_size, ptr, offset, val)                                                       \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset, val)

#define BLOCK_READN_SLM(type, vector_size, ptr, offset)                                                         \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset))
#define BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)                                                   \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset, val)

#define DT_INPUT_BLOCK_READ(ptr, offset)            BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset)          BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)

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

#define DT_FILTER_BLOCK_READ(ptr, offset)           BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset)          BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset)          BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset)          BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset)         BLOCK_READN(FILTER_TYPE, 16, ptr, offset)

// ====================================================================================================================
