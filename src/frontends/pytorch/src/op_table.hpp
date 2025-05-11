// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

const std::unordered_map<std::string, CreatorFunction> get_supported_ops_ts();
const std::unordered_map<std::string, CreatorFunction> get_supported_ops_fx();

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
