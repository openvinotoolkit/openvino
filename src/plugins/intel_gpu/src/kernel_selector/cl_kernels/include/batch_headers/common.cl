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
#define AS_TYPE_PREFIX_fp4e2m1_t _as_
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

#if defined(cl_khr_fp16)
// Float16 Softplus (log(1 + exp(x))) evaluated in float so that exp(x) cannot
// overflow the float16 range (exp(12) is already past the f16 max of ~65504).
// The small-x branch keeps the historical log(exp(x) + 1) expression for
// bit-compatibility; for x >= 20, log(1 + exp(x)) equals x to f16 precision.
inline half softplus_f16(half x) __attribute__((overloadable)) {
    float xf = convert_float(x);
    return convert_half(xf > 20.0f ? xf : log(exp(xf) + 1.0f));
}
inline half4 softplus_f16(half4 x) __attribute__((overloadable)) {
    float4 xf = convert_float4(x);
    return convert_half4(xf > 20.0f ? xf : log(exp(xf) + 1.0f));
}
#endif

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
