// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace GNAPluginNS {

void ConvertToInt16(int16_t *ptr_dst,
                    const float *ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor);

int16_t ConvertFloatToInt16(float src);
int8_t ConvertFloatToInt8(float src);

template<typename T1, typename T2>
inline void UnscaleAndCast(T2 *ptr_dst, T1 *ptr_src, const uint32_t num_rows, const uint32_t num_columns,
                                       const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (uint32_t i = 0; i < num_rows; i++) {
        T1 *ptr_int_row = ptr_src + i * num_columns;
        T2 *ptr_float_row = ptr_dst + i * num_columns;
        for (uint32_t j = 0; j < num_columns; j++) {
            ptr_float_row[j] = static_cast<T2>(ptr_int_row[j]) / scale_factor;
        }
    }
}

}  // namespace GNAPluginNS
