// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "paddlepaddle_frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
using CreatorFunction = std::function<NamedOutputs(const ov::frontend::pdpd::NodeContext&)>;
std::map<std::string, CreatorFunction> get_supported_ops();

}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
