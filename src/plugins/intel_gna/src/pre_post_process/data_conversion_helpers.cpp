// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "data_conversion_helpers.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const size_t num_rows,
                    const size_t num_columns,
                    const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (size_t i = 0; i < num_rows * num_columns; i++) {
        ptr_dst[i] = FloatToInt<int16_t>(ptr_src[i] * scale_factor);
    }
}

}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov
