// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "ngraph/node.hpp"

namespace ngraph {
namespace runtime {
namespace interpreter {
using EvaluatorsMap = std::map<ngraph::NodeTypeInfo,
                               std::function<bool(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::HostTensorVector& outputs,
                                                  const ngraph::HostTensorVector& inputs)>>;
EvaluatorsMap& get_evaluators_map();
}  // namespace interpreter
}  // namespace runtime
}  // namespace ngraph
