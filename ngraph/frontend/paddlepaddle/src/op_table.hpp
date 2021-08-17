// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <ngraph/output_vector.hpp>
#include <string>

#include "node_context.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
using CreatorFunction = std::function<NamedOutputs(const NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph
