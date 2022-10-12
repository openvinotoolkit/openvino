// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing.hpp"

int16_t GNAPluginNS::ConvertFloatToInt16(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > 32767.0) {
        return 32767;
    } else if (value < -32768.0) {
        return -32768;
    }
    return (int16_t)value;
}

int8_t GNAPluginNS::ConvertFloatToInt8(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > 127.0) {
        return 127;
    } else if (value < -128.0) {
        return -128;
    }
    return (int8_t)value;
}

void GNAPluginNS::ConvertToInt16(int16_t *ptr_dst,
                                 const float *ptr_src,
                                 const uint32_t num_rows,
                                 const uint32_t num_columns,
                                 const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (uint32_t i = 0; i < num_rows*num_columns; i++) {
        ptr_dst[i] = GNAPluginNS::ConvertFloatToInt16(ptr_src[i]*scale_factor);
    }
}
