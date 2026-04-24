// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/exception.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {

std::shared_ptr<Node> reorder_axes(const Output<Node>& value, std::vector<size_t> axes_order);
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
