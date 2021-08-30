// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_input.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_value.hpp"
#include "ngraph/output_vector.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"
#include "ngraph/variant.hpp"
#include "openvino/core/node.hpp"

namespace ngraph {

using ov::Node;

namespace runtime {
class HostTensor;
}
using HostTensor = runtime::HostTensor;
using HostTensorPtr = std::shared_ptr<HostTensor>;
using HostTensorVector = std::vector<HostTensorPtr>;

namespace op {

namespace v0 {
class Result;
}
}  // namespace op

using EvaluationContext = std::map<std::string, std::shared_ptr<Variant>>;
using ResultVector = std::vector<std::shared_ptr<ngraph::op::v0::Result>>;

const auto node_validation_failure_loc_string = ov::node_validation_failure_loc_string;
const auto check_single_output_arg = ov::check_single_output_arg;

const auto check_single_output_args = ov::check_single_output_args;

const auto as_output_vector = ov::as_output_vector;
const auto as_node_vector = ov::as_node_vector;
const auto as_result_vector = ov::as_result_vector;

/// Alias useful for cloning
using NodeMap = std::unordered_map<ngraph::Node*, std::shared_ptr<ngraph::Node>>;

/// Nodes are the backbone of the graph of Value dataflow. Every node has
/// zero or more nodes as arguments and one value, which is either a tensor
/// or a (possibly empty) tuple of values.
using ov::NodeValidationFailure;

using NodeTypeInfo = Node::type_info_t;

// Like an Output but with a Node* instead of a shared_ptr<Node>
using ov::RawNodeOutput;

using RawNodeOutputMap = std::map<RawNodeOutput, Output<Node>>;

using ov::check_new_args_count;

#define NGRAPH_RTTI_DECLARATION OPENVINO_RTTI_DECLARATION

#define NGRAPH_RTTI_DEFINITION(...) OPENVINO_RTTI_DEFINITION(__VA_ARGS__)

}  // namespace ngraph
