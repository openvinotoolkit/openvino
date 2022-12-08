// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define DECLARE_LOWER_BOUND(Name, Type, ValType, GetIndex)                                        \
    inline Type* FUNC(Name)(const Type* data, uint first_index, uint last_index, ValType value) { \
        uint count = last_index - first_index;                                                    \
        while (count > 0) {                                                                       \
            const uint step = count / 2;                                                          \
            const uint middle_index = first_index + step;                                         \
            if (data[GetIndex(middle_index)] < value) {                                           \
                first_index = middle_index + 1;                                                   \
                count -= step + 1;                                                                \
            } else {                                                                              \
                count = step;                                                                     \
            }                                                                                     \
        }                                                                                         \
        return first_index;                                                                       \
    }

#define DECLARE_UPPER_BOUND(Name, Type, ValType, GetIndex)                                        \
    inline Type* FUNC(Name)(const Type* data, uint first_index, uint last_index, ValType value) { \
        uint count = last_index - first_index;                                                    \
        while (count > 0) {                                                                       \
            const uint step = count / 2;                                                          \
            const uint middle_index = first_index + step;                                         \
            if (value < data[GetIndex(middle_index)]) {                                           \
                count = step;                                                                     \
            } else {                                                                              \
                first_index = middle_index + 1;                                                   \
                count -= step + 1;                                                                \
            }                                                                                     \
        }                                                                                         \
        return first_index;                                                                       \
    }
