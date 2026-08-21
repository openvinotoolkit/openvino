// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>

namespace kernel_selector {

inline bool is_integral_ratio(size_t lhs, size_t rhs) {
    return lhs != 0 && rhs != 0 && (lhs % rhs == 0 || rhs % lhs == 0);
}

inline bool is_integral_upsampling_ratio(size_t output, size_t input) {
    return input != 0 && output >= input && output % input == 0;
}

}  // namespace kernel_selector
