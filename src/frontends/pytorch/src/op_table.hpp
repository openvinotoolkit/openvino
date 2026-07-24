// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

namespace op {
OutputVector translate__nested_tensor_from_mask(const NodeContext& context);
}  // namespace op

const std::unordered_map<std::string, CreatorFunction> get_supported_ops_ts();
const std::unordered_map<std::string, CreatorFunction> get_supported_ops_fx();

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
