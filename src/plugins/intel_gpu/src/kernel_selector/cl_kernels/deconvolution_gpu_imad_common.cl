// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define VEC_TO_ARR_1(var, arr, idx)             \
    arr[idx] = var
#define VEC_TO_ARR_2(vec, arr, idx)             \
    VEC_TO_ARR_1((vec).lo, arr, idx);           \
    VEC_TO_ARR_1((vec).hi, arr, (idx) + 1)
#define VEC_TO_ARR_4(vec, arr, idx)             \
    VEC_TO_ARR_2((vec).lo, arr, idx);           \
    VEC_TO_ARR_2((vec).hi, arr, (idx) + 2)
#define VEC_TO_ARR_8(vec, arr, idx)             \
    VEC_TO_ARR_4((vec).lo, arr, idx);           \
    VEC_TO_ARR_4((vec).hi, arr, (idx) + 4)
#define VEC_TO_ARR_16(vec, arr, idx)            \
    VEC_TO_ARR_8((vec).lo, arr, idx);           \
    VEC_TO_ARR_8((vec).hi, arr, (idx) + 8)

#define ARR_TO_VEC_1(arr, var, idx)             \
    var = arr[idx]
#define ARR_TO_VEC_2(arr, vec, idx)             \
    ARR_TO_VEC_1(arr, (vec).lo, idx);           \
    ARR_TO_VEC_1(arr, (vec).hi, (idx) + 1)
#define ARR_TO_VEC_4(arr, vec, idx)             \
    ARR_TO_VEC_2(arr, (vec).lo, idx);           \
    ARR_TO_VEC_2(arr, (vec).hi, (idx) + 2)
#define ARR_TO_VEC_8(arr, vec, idx)             \
    ARR_TO_VEC_4(arr, (vec).lo, idx);           \
    ARR_TO_VEC_4(arr, (vec).hi, (idx) + 4)
#define ARR_TO_VEC_16(arr, vec, idx)            \
    ARR_TO_VEC_8(arr, (vec).lo, idx);           \
    ARR_TO_VEC_8(arr, (vec).hi, (idx) + 8)

#define DECLARE_LOAD_CONTINOUS_16(name, type)                                                                       \
inline void FUNC(name)(const __global type* src, uint offset, uint size, type* dst) {                               \
    uint i = 0;                                                                                                     \
    for (; i + 16 <= size; i += 16) {                                                                               \
        MAKE_VECTOR_TYPE(type, 16) tmp = ((const __global MAKE_VECTOR_TYPE(type, 16)*)(src + offset + i))[0];       \
        VEC_TO_ARR_16(tmp, dst, i);                                                                                 \
    }                                                                                                               \
    if (size % 16 >= 8) {                                                                                           \
        MAKE_VECTOR_TYPE(type, 8) tmp = ((const __global MAKE_VECTOR_TYPE(type, 8)*)(src + offset + i))[0];         \
        VEC_TO_ARR_8(tmp, dst, i);                                                                                  \
        i += 8;                                                                                                     \
    }                                                                                                               \
    if (size % 8 >= 4) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 4) tmp = ((const __global MAKE_VECTOR_TYPE(type, 4)*)(src + offset + i))[0];         \
        VEC_TO_ARR_4(tmp, dst, i);                                                                                  \
        i += 4;                                                                                                     \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp = ((const __global MAKE_VECTOR_TYPE(type, 2)*)(src + offset + i))[0];         \
        VEC_TO_ARR_2(tmp, dst, i);                                                                                  \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        dst[i] = src[offset + i];                                                                                   \
    }                                                                                                               \
}

#define DECLARE_LOAD_CONTINOUS_4(name, type)                                                                        \
inline void FUNC(name)(const __global type* src, uint offset, uint size, type* dst) {                               \
    uint i = 0;                                                                                                     \
    for (; i + 4 <= size; i += 4) {                                                                                 \
        MAKE_VECTOR_TYPE(type, 4) tmp = ((const __global MAKE_VECTOR_TYPE(type, 4)*)(src + offset + i))[0];         \
        VEC_TO_ARR_4(tmp, dst, i);                                                                                  \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp = ((const __global MAKE_VECTOR_TYPE(type, 2)*)(src + offset + i))[0];         \
        VEC_TO_ARR_2(tmp, dst, i);                                                                                  \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        dst[i] = src[offset + i];                                                                                   \
    }                                                                                                               \
}

#define DECLARE_STORE_BLOCK_16(name, type)                                                                          \
inline void FUNC(name)(__global type* dst, uint offset, uint size, type* src) {                                     \
    uint i = 0;                                                                                                     \
    const uint sg_size = get_max_sub_group_size();                                                                  \
    for (; i + 16 <= size; i += 16) {                                                                               \
        MAKE_VECTOR_TYPE(type, 16) tmp;                                                                             \
        ARR_TO_VEC_16(src, tmp, i);                                                                                 \
        BLOCK_WRITEN(type, 16, dst, offset + i * sg_size, tmp);                                                     \
    }                                                                                                               \
    if (size % 16 >= 8) {                                                                                           \
        MAKE_VECTOR_TYPE(type, 8) tmp;                                                                              \
        ARR_TO_VEC_8(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 8, dst, offset + i * sg_size, tmp);                                                      \
        i += 8;                                                                                                     \
    }                                                                                                               \
    if (size % 8 >= 4) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 4) tmp;                                                                              \
        ARR_TO_VEC_4(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 4, dst, offset + i * sg_size, tmp);                                                      \
        i += 4;                                                                                                     \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp;                                                                              \
        ARR_TO_VEC_2(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 2, dst, offset + i * sg_size, tmp);                                                      \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        type tmp = src[i];                                                                                          \
        BLOCK_WRITEN(type, 1, dst, offset + i * sg_size, tmp);                                                      \
    }                                                                                                               \
}

#define DECLARE_STORE_BLOCK_8(name, type)                                                                           \
inline void FUNC(name)(__global type* dst, uint offset, uint size, type* src) {                                     \
    uint i = 0;                                                                                                     \
    const uint sg_size = get_max_sub_group_size();                                                                  \
    for (; i + 8 <= size; i += 8) {                                                                                 \
        MAKE_VECTOR_TYPE(type, 8) tmp;                                                                              \
        ARR_TO_VEC_8(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 8, dst, offset + i * sg_size, tmp);                                                      \
    }                                                                                                               \
    if (size % 8 >= 4) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 4) tmp;                                                                              \
        ARR_TO_VEC_4(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 4, dst, offset + i * sg_size, tmp);                                                      \
        i += 4;                                                                                                     \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp;                                                                              \
        ARR_TO_VEC_2(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 2, dst, offset + i * sg_size, tmp);                                                      \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        type tmp = src[i];                                                                                          \
        BLOCK_WRITEN(type, 1, dst, offset + i * sg_size, tmp);                                                      \
    }                                                                                                               \
}

#define DECLARE_STORE_BLOCK_4(name, type)                                                                           \
inline void FUNC(name)(__global type* dst, uint offset, uint size, type* src) {                                     \
    uint i = 0;                                                                                                     \
    const uint sg_size = get_max_sub_group_size();                                                                  \
    for (; i + 4 <= size; i += 4) {                                                                                 \
        MAKE_VECTOR_TYPE(type, 4) tmp;                                                                              \
        ARR_TO_VEC_4(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 4, dst, offset + i * sg_size, tmp);                                                      \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp;                                                                              \
        ARR_TO_VEC_2(src, tmp, i);                                                                                  \
        BLOCK_WRITEN(type, 2, dst, offset + i * sg_size, tmp);                                                      \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        type tmp = src[i];                                                                                          \
        BLOCK_WRITEN(type, 1, dst, offset + i * sg_size, tmp);                                                      \
    }                                                                                                               \
}

#define DECLARE_READ_BLOCK_16(name, type)                                                                           \
inline void FUNC(name)(const __global type* src, uint offset, uint size, type* dst) {                               \
    uint i = 0;                                                                                                     \
    const uint sg_size = get_max_sub_group_size();                                                                  \
    for (; i + 16 <= size; i += 16) {                                                                               \
        MAKE_VECTOR_TYPE(type, 16) tmp = BLOCK_READN(type, 16, src, offset + i * sg_size);                          \
        VEC_TO_ARR_16(tmp, dst, i);                                                                                 \
    }                                                                                                               \
    if (size % 16 >= 8) {                                                                                           \
        MAKE_VECTOR_TYPE(type, 8) tmp = BLOCK_READN(type, 8, src, offset + i * sg_size);                            \
        VEC_TO_ARR_8(tmp, dst, i);                                                                                  \
        i += 8;                                                                                                     \
    }                                                                                                               \
    if (size % 8 >= 4) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 4) tmp = BLOCK_READN(type, 4, src, offset + i * sg_size);                            \
        VEC_TO_ARR_4(tmp, dst, i);                                                                                  \
        i += 4;                                                                                                     \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp = BLOCK_READN(type, 2, src, offset + i * sg_size);                            \
        VEC_TO_ARR_2(tmp, dst, i);                                                                                  \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        type tmp = BLOCK_READN(type, 1, src, offset + i * sg_size);                                                 \
        dst[i] = tmp;                                                                                               \
    }                                                                                                               \
}

#define DECLARE_READ_BLOCK_8(name, type)                                                                            \
inline void FUNC(name)(const __global type* src, uint offset, uint size, type* dst) {                               \
    uint i = 0;                                                                                                     \
    const uint sg_size = get_max_sub_group_size();                                                                  \
    for (; i + 8 <= size; i += 8) {                                                                                 \
        MAKE_VECTOR_TYPE(type, 8) tmp = BLOCK_READN(type, 8, src, offset + i * sg_size);                            \
        VEC_TO_ARR_8(tmp, dst, i);                                                                                  \
    }                                                                                                               \
    if (size % 8 >= 4) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 4) tmp = BLOCK_READN(type, 4, src, offset + i * sg_size);                            \
        VEC_TO_ARR_4(tmp, dst, i);                                                                                  \
        i += 4;                                                                                                     \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp = BLOCK_READN(type, 2, src, offset + i * sg_size);                            \
        VEC_TO_ARR_2(tmp, dst, i);                                                                                  \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        type tmp = BLOCK_READN(type, 1, src, offset + i * sg_size);                                                 \
        dst[i] = tmp;                                                                                               \
    }                                                                                                               \
}

#define DECLARE_READ_BLOCK_4(name, type)                                                                            \
inline void FUNC(name)(const __global type* src, uint offset, uint size, type* dst) {                               \
    uint i = 0;                                                                                                     \
    const uint sg_size = get_max_sub_group_size();                                                                  \
    for (; i + 4 <= size; i += 4) {                                                                                 \
        MAKE_VECTOR_TYPE(type, 4) tmp = BLOCK_READN(type, 4, src, offset + i * sg_size);                            \
        VEC_TO_ARR_4(tmp, dst, i);                                                                                  \
    }                                                                                                               \
    if (size % 4 >= 2) {                                                                                            \
        MAKE_VECTOR_TYPE(type, 2) tmp = BLOCK_READN(type, 2, src, offset + i * sg_size);                            \
        VEC_TO_ARR_2(tmp, dst, i);                                                                                  \
        i += 2;                                                                                                     \
    }                                                                                                               \
    if (size % 2 == 1) {                                                                                            \
        type tmp = BLOCK_READN(type, 1, src, offset + i * sg_size);                                                 \
        dst[i] = tmp;                                                                                               \
    }                                                                                                               \
}
