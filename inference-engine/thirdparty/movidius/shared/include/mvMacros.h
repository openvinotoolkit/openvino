/*
* Copyright 2017-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

#ifndef MVMACROS_H__
#define MVMACROS_H__

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((!(sizeof(x) % sizeof(0[x])))))
#ifndef MIN
#define MIN(a,b)                                \
    ({ __typeof__ (a) _a = (a);                 \
        __typeof__ (b) _b = (b);                \
        _a < _b ? _a : _b; })
#endif
#ifndef MAX
#define MAX(a,b)                                \
    ({ __typeof__ (a) _a = (a);                 \
        __typeof__ (b) _b = (b);                \
        _a > _b ? _a : _b; })
#endif
/// @brief Aligns a pointer or number to a power of 2 value given
/// @param[in] x number or pointer to be aligned
/// @param[in] a value to align to (must be power of 2)
/// @returns the aligned value
#if (defined(_WIN32) || defined(_WIN64) )
#define ALIGN_UP_UINT32(x, a)   ((uint32_t)(((uint32_t)(x) + a - 1) & (~(a-1))))
#define ALIGN_UP_INT32(x, a)   ((int32_t)(((uint32_t)(x) + a - 1) & (~(a-1))))
#define ALIGN_UP(x, a) ALIGN_UP_UINT32(x,a)
#else
#define ALIGN_UP(x, a)   ((typeof(x))(((uint32_t)(x) + a - 1) & (~(a-1))))
#define ALIGN_DOWN(x, a) ((typeof(x))(((uint32_t)(x)) & (~(a-1))) )
#define ALIGN_UP_UINT32(_x, _a)   ALIGN_UP(_x, _a)
#define ALIGN_UP_INT32(_x, _a)   ALIGN_UP(_x, _a)
#endif
/// @brief Aligns a integernumber to any value given
/// @param[in] x integer number to be aligned
/// @param[in] a value to align to
/// @returns the aligned value
#ifndef ROUND_UP
#define ROUND_UP(x, a)   ((__typeof__(x))((((uint32_t)(x) + a - 1) / a) * a))
#endif
#define ROUND_DOWN(x, a) ((__typeof__(x))(((uint32_t)(x) / a + 0) * a))

#if defined(__GNUC__) || defined(__sparc_v8__)
#define ATTR_UNUSED __attribute__((unused))
#else
#define ATTR_UNUSED
#endif

#endif

