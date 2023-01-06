// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.cl"

#ifdef cl_intel_subgroups
#define _sub_group_shuffle(v, c) intel_sub_group_shuffle(v, c)
#define _sub_group_shuffle_up(c, n, d) intel_sub_group_shuffle_up(c, n, d)
#define _sub_group_shuffle_down(c, n, d) intel_sub_group_shuffle_down(c, n, d)
#elif (__OPENCL_C_VERSION__ >= 200)

// The spec for intel_subgroup_shuffle says that index (c) need not be the same value for all work-items in
// a subgroup while sub_group_broadcast requires that.
// However, most of our kernels uses shuffle in a way that produces same index for all work-items,
// so for now we use this solution.
// In case of accuracy issues we may switch to something like this:
// #define MAX_SG_SIZE 32
// #define DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)
// inline type _sub_group_shuffle(type v, uint c) __attribute__((overloadable)) {
//     type vals[MAX_SG_SIZE];
//     for (size_t i = 0; i < get_max_sub_group_size(); i++) {
//         vals[i] = AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v), i));
//     }
//     return vals[c];
// }

#define DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)                                               \
inline type _sub_group_shuffle(type v, uint c) __attribute__((overloadable)) {                    \
    return AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v), c));                          \
}

#define DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type)                                               \
inline CAT(type, 2) _sub_group_shuffle(CAT(type, 2) v, uint c) __attribute__((overloadable)) {    \
    return (CAT(type, 2))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)));      \
}

#define DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type)                                               \
inline CAT(type, 4) _sub_group_shuffle(CAT(type, 4) v, uint c) __attribute__((overloadable)) {    \
   return (CAT(type, 4))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)));       \
}

#define DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type)                                               \
inline CAT(type, 8) _sub_group_shuffle(CAT(type, 8) v, uint c) __attribute__((overloadable)) {    \
   return (CAT(type, 8))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c)));       \
}

#define DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type)                                              \
inline CAT(type, 16) _sub_group_shuffle(CAT(type, 16) v, uint c) __attribute__((overloadable)) {  \
   return (CAT(type, 16))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s8), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s9), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sa), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sb), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sc), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sd), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.se), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sf), c)));      \
}


#define DECLARE_SUB_GROUP_SHUFFLE(type)    \
    DECLARE_SUB_GROUP_SHUFFLE1(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE2(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE4(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE8(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE16(type, type)

#define DECLARE_SUB_GROUP_SHUFFLE_CASTED(type, cast_type) \
    DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type)


DECLARE_SUB_GROUP_SHUFFLE(int)
DECLARE_SUB_GROUP_SHUFFLE(uint)
DECLARE_SUB_GROUP_SHUFFLE(float)

#if defined(cl_khr_fp16)
    DECLARE_SUB_GROUP_SHUFFLE(half)
    DECLARE_SUB_GROUP_SHUFFLE_CASTED(short, half)
    DECLARE_SUB_GROUP_SHUFFLE_CASTED(ushort, half)
#endif

#endif
