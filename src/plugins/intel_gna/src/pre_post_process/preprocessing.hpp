// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "gna_data_types.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const size_t num_rows,
                    const size_t num_columns,
                    const float scale_factor);

template <typename T>
T FloatToInt(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > static_cast<float>(std::numeric_limits<T>::max())) {
        return std::numeric_limits<T>::max();
    } else if (value < static_cast<float>(std::numeric_limits<T>::min())) {
        return std::numeric_limits<T>::min();
    }
    return static_cast<T>(value);
}
}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov