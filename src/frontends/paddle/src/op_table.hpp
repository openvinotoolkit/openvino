// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/frontend/paddle/node_context.hpp"

namespace ov::frontend::paddle {
using CreatorFunction = std::function<NamedOutputs(const NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

}  // namespace ov::frontend::paddle
