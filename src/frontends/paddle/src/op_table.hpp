// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
using CreatorFunction = std::function<NamedOutputs(const NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
