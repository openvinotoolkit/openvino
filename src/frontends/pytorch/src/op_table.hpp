// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
using CreatorFunction = std::function<OutputVector(NodeContext&)>;

const std::map<std::string, CreatorFunction> get_supported_ops();

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
