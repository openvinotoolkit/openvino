// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define DECLARE_LOWER_BOUND(Name, Type, ValType)                        \
    inline Type* FUNC(Name)(Type * first, Type * last, ValType value) { \
        uint count = last - first;                                      \
        while (count > 0) {                                             \
            const uint step = count / 2;                                \
            const Type* middle = first + step;                          \
            if (*middle < value) {                                      \
                first = middle + 1;                                     \
                count -= step + 1;                                      \
            } else {                                                    \
                count = step;                                           \
            }                                                           \
        }                                                               \
        return first;                                                   \
    }

#define DECLARE_UPPER_BOUND(Name, Type, ValType)                        \
    inline Type* FUNC(Name)(Type * first, Type * last, ValType value) { \
        uint count = last - first;                                      \
        while (count > 0) {                                             \
            const uint step = count / 2;                                \
            const Type* middle = first + step;                          \
            if (value < *middle) {                                      \
                count = step;                                           \
            } else {                                                    \
                first = middle + 1;                                     \
                count -= step + 1;                                      \
            }                                                           \
        }                                                               \
        return first;                                                   \
    }
