// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "node_context.hpp"

namespace ov {
namespace frontend {
namespace paddlepaddle {
using CreatorFunction = std::function<NamedOutputs(const NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

}  // namespace paddlepaddle
}  // namespace frontend
}  // namespace ov
