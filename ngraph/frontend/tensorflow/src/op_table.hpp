// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <functional>

#include "utils.h"
#include <default_opset.h>
#include <ngraph/output_vector.hpp>
#include <tensorflow_frontend/node_context.hpp>

namespace tensorflow {
namespace ngraph_bridge {
using OutPortName = std::string;
using NamedOutputs = ngraph::OutputVector;
using CreatorFunction = std::function<NamedOutputs(const ngraph::frontend::tensorflow::detail::NodeContext&)>;

std::map<const std::string, CreatorFunction> get_supported_ops();

}  // namespace ngraph_bridge
}  // namespace tensorflow
