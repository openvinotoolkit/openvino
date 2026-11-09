// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/paddle/node_context.hpp"

namespace ov::frontend::paddle {

Output<Node> get_tensor_list(const OutputVector& node);
Output<Node> get_tensor_safe(const Output<Node>& node);
}  // namespace ov::frontend::paddle
