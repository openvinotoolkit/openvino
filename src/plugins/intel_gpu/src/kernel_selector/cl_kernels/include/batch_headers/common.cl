// Copyright (C) 2018-2024 Intel Corporation
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
#define TYPE_SIZE_ulong  8
#define TYPE_SIZE_long   8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif

float convert_bf16_to_float(const ushort in){
    uint u = 0;
    //sign
    if ( (in>>15) ) { 
        u = 1 << 31;
    }
    //exponent
    u += ( ( (in >> 7) & 0b11111111)) << 23;
    //fraction 
    u += (in & 0b1111111) << 16;
    float* f = &u;
    printf("I ret %f %e u is  %u in is %u or %x\n", *f, *f, u, in, in);
    return *f;
}

ushort convert_bf16(const float in_f){
    uint* in = &in_f;
    ushort u = 0;
    if ( (*in>>31) ) { 
        u = 1 << 15;
    }
    //exponent
    u += ( ( (*in >> 23) & 0b11111111)) << 7;
    //fraction
    u += (*in >> 16) & 0b1111111;
    return u;
}