// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "ngraph/node.hpp"

std::vector<float> get_floats(const std::shared_ptr<ngraph::HostTensor>& input, const ngraph::Shape& shape);

std::vector<int64_t> get_integers(const std::shared_ptr<ngraph::HostTensor>& input, const ngraph::Shape& shape);

std::vector<int64_t> get_signal_size(const std::vector<std::shared_ptr<ngraph::HostTensor>>& inputs,
                                     size_t num_of_axes);

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
