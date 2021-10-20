// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {

template <typename NodeType>
class Output;

class Node;

namespace op {
namespace v0 {

class Result;
class Parameter;

}  // namespace v0
class Sink;
}  // namespace op

using NodeVector = std::vector<std::shared_ptr<Node>>;
using ConstNodeVector = std::vector<std::shared_ptr<const Node>>;
using OutputVector = std::vector<Output<Node>>;
using ConstOutputVector = std::vector<Output<const Node>>;
using ParameterVector = std::vector<std::shared_ptr<op::v0::Parameter>>;
using ConstParameterVector = std::vector<std::shared_ptr<const op::v0::Parameter>>;
using ResultVector = std::vector<std::shared_ptr<ov::op::v0::Result>>;
using ConstResultVector = std::vector<std::shared_ptr<const ov::op::v0::Result>>;
using SinkVector = std::vector<std::shared_ptr<op::Sink>>;
using ConstSinkVector = std::vector<std::shared_ptr<const op::Sink>>;

OPENVINO_API
OutputVector as_output_vector(const NodeVector& args);
OPENVINO_API
NodeVector as_node_vector(const OutputVector& values);
/// Returns a ResultVector referencing values.
OPENVINO_API
ResultVector as_result_vector(const OutputVector& values);
}  // namespace ov
