// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
Output<Node> make_broadcast(const Output<Node>& node, const Shape& target_shape, const AxisSet& broadcast_axes);

Output<Node> make_broadcast(const Output<Node>& node, const Shape& target_shape, std::size_t start_match_axis);
}  // namespace util
}  // namespace op
}  // namespace ov
