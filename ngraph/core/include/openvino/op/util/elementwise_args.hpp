// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
std::tuple<element::Type, Shape> validate_and_infer_elementwise_args(
    Node* node,
    const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());
}
}  // namespace op
}  // namespace ov
