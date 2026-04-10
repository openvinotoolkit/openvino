// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#   define ASSUME_HINT(x) __builtin_assume(x)
#else
#   define ASSUME_HINT(x) do { } while (0)
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b)      ((a) < (b) ? (a) : (b))
#define MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE_PREFIX_uchar as_
#define AS_TYPE_PREFIX_char as_
#define AS_TYPE_PREFIX_fp8e5m2_t _as_
#define AS_TYPE_PREFIX_fp8e4m3_t _as_
#define AS_TYPE_PREFIX_fp8e8m0_t _as_
#define AS_TYPE_PREFIX_ushort as_
#define AS_TYPE_PREFIX_short as_
#define AS_TYPE_PREFIX_half as_
#define AS_TYPE_PREFIX_int as_
#define AS_TYPE_PREFIX_uint as_
#define AS_TYPE_PREFIX_float as_
#define AS_TYPE_PREFIX_ulong as_
#define AS_TYPE_PREFIX_long as_

#define AS_TYPE_EXT(type, val, src_type) CAT(CAT(AS_TYPE_PREFIX_, src_type), type)(val)
#define AS_TYPE(type, val) CAT(as_, type)(val)

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
#define TYPE_SIZE_uchar  1
#define TYPE_SIZE_char   1
#define TYPE_SIZE_fp8e5m2_t 1
#define TYPE_SIZE_fp8e4m3_t 1
#define TYPE_SIZE_fp8e8m0_t 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short  2
#define TYPE_SIZE_half   2
#define TYPE_SIZE_int    4
#define TYPE_SIZE_uint   4
#define TYPE_SIZE_float  4
#define TYPE_SIZE_ulong  8
#define TYPE_SIZE_long   8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif

#define DEFINE_VECTOR_CONVERT(target_type, source_type, vector_size) \
MAKE_VECTOR_TYPE(target_type, vector_size) __attribute__((overloadable)) _convert_##target_type##vector_size(MAKE_VECTOR_TYPE(source_type, vector_size) val) { \
    return CAT(convert_, CAT(target_type, vector_size))(val); \
}

#define DEFINE_VECTOR_IDENTITY(type, vector_size) \
MAKE_VECTOR_TYPE(type, vector_size) __attribute__((overloadable)) _convert_##type##vector_size(MAKE_VECTOR_TYPE(type, vector_size) val) { \
    return val; \
}

#define DEFINE_ALL_VECTOR_CONVERTS(target_type, source_type) \
DEFINE_VECTOR_CONVERT(target_type, source_type, 2) \
DEFINE_VECTOR_CONVERT(target_type, source_type, 3) \
DEFINE_VECTOR_CONVERT(target_type, source_type, 4) \
DEFINE_VECTOR_CONVERT(target_type, source_type, 8) \
DEFINE_VECTOR_CONVERT(target_type, source_type, 16)

#define DEFINE_ALL_VECTOR_IDENTITY(type) \
DEFINE_VECTOR_IDENTITY(type, 2) \
DEFINE_VECTOR_IDENTITY(type, 3) \
DEFINE_VECTOR_IDENTITY(type, 4) \
DEFINE_VECTOR_IDENTITY(type, 8) \
DEFINE_VECTOR_IDENTITY(type, 16)

float __attribute__((overloadable)) _convert_float(char val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(uchar val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(short val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(ushort val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(int val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(uint val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(long val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(ulong val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(half val) { return convert_float(val); }
float __attribute__((overloadable)) _convert_float(float val) { return val; }

DEFINE_ALL_VECTOR_CONVERTS(float, char)
DEFINE_ALL_VECTOR_CONVERTS(float, uchar)
DEFINE_ALL_VECTOR_CONVERTS(float, short)
DEFINE_ALL_VECTOR_CONVERTS(float, ushort)
DEFINE_ALL_VECTOR_CONVERTS(float, int)
DEFINE_ALL_VECTOR_CONVERTS(float, uint)
DEFINE_ALL_VECTOR_CONVERTS(float, long)
DEFINE_ALL_VECTOR_CONVERTS(float, ulong)
DEFINE_ALL_VECTOR_CONVERTS(float, half)
DEFINE_ALL_VECTOR_IDENTITY(float)

half __attribute__((overloadable)) _convert_half(char val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(uchar val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(short val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(ushort val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(int val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(uint val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(long val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(ulong val) { return convert_half(val); }
half __attribute__((overloadable)) _convert_half(half val) { return val; }
half __attribute__((overloadable)) _convert_half(float val) { return convert_half(val); }

DEFINE_ALL_VECTOR_CONVERTS(half, char)
DEFINE_ALL_VECTOR_CONVERTS(half, uchar)
DEFINE_ALL_VECTOR_CONVERTS(half, short)
DEFINE_ALL_VECTOR_CONVERTS(half, ushort)
DEFINE_ALL_VECTOR_CONVERTS(half, int)
DEFINE_ALL_VECTOR_CONVERTS(half, uint)
DEFINE_ALL_VECTOR_CONVERTS(half, long)
DEFINE_ALL_VECTOR_CONVERTS(half, ulong)
DEFINE_ALL_VECTOR_CONVERTS(half, float)
DEFINE_ALL_VECTOR_IDENTITY(half)
