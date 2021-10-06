// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <ngraph/output_vector.hpp>
#include <string>

#include "ngraph_conversions.hpp"
#include "node_context.hpp"
#include "utils.hpp"

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {
using CreatorFunction = std::function<::ngraph::OutputVector(const ::ngraph::frontend::tf::NodeContext&)>;

const std::map<const std::string, const CreatorFunction> get_supported_ops();
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
