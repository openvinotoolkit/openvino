// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<Node> make_broadcast(const Output<Node>& node,
                                     const Shape& target_shape,
                                     const AxisSet& broadcast_axes);

std::shared_ptr<Node> make_broadcast(const Output<Node>& node, const Shape& target_shape, std::size_t start_match_axis);
}  // namespace utils
}  // namespace test
}  // namespace ov