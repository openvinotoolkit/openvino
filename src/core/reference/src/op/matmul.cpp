// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/matmul.hpp"

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace ov {
namespace reference {
namespace details {
std::vector<size_t> get_transpose_order(const Shape& input_shape) {
    size_t rank = input_shape.size();
    OPENVINO_ASSERT(rank > 1, "Invalid input for transpose");
    std::vector<size_t> axes_order(rank);
    std::iota(axes_order.begin(), axes_order.end(), 0);
    std::swap(axes_order[rank - 1], axes_order[rank - 2]);
    return axes_order;
}
}  // namespace details
}  // namespace reference
}  // namespace ov
