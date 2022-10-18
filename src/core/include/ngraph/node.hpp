// Copyright (C) 2018-2022 Intel Corporation
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
#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace v0 {
class Result;
}
}  // namespace op
}  // namespace ov
namespace ngraph {

using ov::Node;

namespace runtime {
class HostTensor;
}
using HostTensor = runtime::HostTensor;
using HostTensorPtr = std::shared_ptr<HostTensor>;
using HostTensorVector = std::vector<HostTensorPtr>;
using TensorLabel = std::vector<size_t>;
using TensorLabelVector = std::vector<TensorLabel>;

namespace op {

namespace v0 {
using ov::op::v0::Result;
}
}  // namespace op

using EvaluationContext = ov::EvaluationContext;
using ResultVector = std::vector<std::shared_ptr<ngraph::op::v0::Result>>;

const auto node_validation_failure_loc_string = ov::node_validation_failure_loc_string;

NGRAPH_API
const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node, size_t i);
NGRAPH_API
const NodeVector& check_single_output_args(const NodeVector& args);

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

/// Helper macro that puts necessary declarations of RTTI block inside a class definition.
/// Should be used in the scope of class that requires type identification besides one provided by
/// C++ RTTI.
/// Recommended to be used for all classes that are inherited from class ov::Node to enable
/// pattern
/// matching for them. Accepts necessary type identification details like type of the operation,
/// version and optional parent class.
///
/// Applying this macro within a class definition provides declaration of type_info static
/// constant for backward compatibility with old RTTI definition for Node,
/// static function get_type_info_static which returns a reference to an object that is equal to
/// type_info but not necessary to the same object, and get_type_info virtual function that
/// overrides Node::get_type_info and returns a reference to the same object that
/// get_type_info_static gives.
///
/// Use this macro as a public part of the class definition:
///
///     class MyOp : public Node
///     {
///         public:
///             // Don't use Node as a parent for type_info, it doesn't have any value and
///             prohibited
///             NGRAPH_RTTI_DECLARATION;
///
///             ...
///     };
///
///     class MyInheritedOp : public MyOp
///     {
///         public:
///             NGRAPH_RTTI_DECLARATION;
///
///             ...
///     };
///
/// To complete type identification for a class, use NGRAPH_RTTI_DEFINITION.
///
#ifdef OPENVINO_STATIC_LIBRARY
#    define NGRAPH_RTTI_DECLARATION                                        \
        const ::ngraph::Node::type_info_t& get_type_info() const override; \
        static const ::ngraph::Node::type_info_t& get_type_info_static()
#    define _NGRAPH_RTTI_DEFINITION_COMMON(CLASS)                         \
        const ::ngraph::Node::type_info_t& CLASS::get_type_info() const { \
            return get_type_info_static();                                \
        }
#else
#    define NGRAPH_RTTI_DECLARATION                                        \
        static const ::ngraph::Node::type_info_t type_info;                \
        const ::ngraph::Node::type_info_t& get_type_info() const override; \
        static const ::ngraph::Node::type_info_t& get_type_info_static()
#    define _NGRAPH_RTTI_DEFINITION_COMMON(CLASS)                         \
        const ::ngraph::Node::type_info_t& CLASS::get_type_info() const { \
            return get_type_info_static();                                \
        }                                                                 \
        const ::ngraph::Node::type_info_t CLASS::type_info = CLASS::get_type_info_static()
#endif

#define _NGRAPH_RTTI_DEFINITION_WITH_PARENT(CLASS, TYPE_NAME, _VERSION_INDEX, PARENT_CLASS)               \
    const ::ngraph::Node::type_info_t& CLASS::get_type_info_static() {                                    \
        static const ::ngraph::Node::type_info_t type_info_static{TYPE_NAME,                              \
                                                                  static_cast<uint64_t>(_VERSION_INDEX),  \
                                                                  &PARENT_CLASS::get_type_info_static()}; \
        return type_info_static;                                                                          \
    }                                                                                                     \
    _NGRAPH_RTTI_DEFINITION_COMMON(CLASS)

#define _NGRAPH_RTTI_DEFINITION_NO_PARENT(CLASS, TYPE_NAME, _VERSION_INDEX)                                          \
    const ::ngraph::Node::type_info_t& CLASS::get_type_info_static() {                                               \
        static const ::ngraph::Node::type_info_t type_info_static{TYPE_NAME, static_cast<uint64_t>(_VERSION_INDEX)}; \
        return type_info_static;                                                                                     \
    }                                                                                                                \
    _NGRAPH_RTTI_DEFINITION_COMMON(CLASS)
#define NGRAPH_RTTI_DEFINITION(...)                                                               \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR(__VA_ARGS__,                         \
                                                             _NGRAPH_RTTI_DEFINITION_WITH_PARENT, \
                                                             _NGRAPH_RTTI_DEFINITION_NO_PARENT)(__VA_ARGS__))

}  // namespace ngraph
