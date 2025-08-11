// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "traits.hpp"

#include "openvino/op/util/op_types.hpp"

namespace ov {
namespace npuw {

bool partitioning::traits::is_tiny_shape(const ov::Shape& shape) {
    const auto total = std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());

    // The initial check
    const std::size_t TINY_THRESHOLD = 16;
    if ((shape.size() == 0 || (shape.size() == 1 && shape[0] <= TINY_THRESHOLD)) || (total <= TINY_THRESHOLD)) {
        return true;
    }
    return false;
}

bool partitioning::traits::is_tiny_scalar(const std::shared_ptr<ov::Node>& node) {
    if (!ov::op::util::is_constant(node)) {
        return false;
    }
    return is_tiny_shape(node->output(0).get_shape());
}

}  // namespace npuw
}  // namespace ov
