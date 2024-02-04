// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/common.hpp"

#include <onnx/onnx_pb.h>  // onnx types

#include "onnx_framework_node.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::op;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace common {
const ov::element::Type& get_ov_element_type(int64_t onnx_type) {
    switch (onnx_type) {
    case TensorProto_DataType::TensorProto_DataType_BOOL:
        return ov::element::boolean;
    case TensorProto_DataType::TensorProto_DataType_DOUBLE:
        return ov::element::f64;
    case TensorProto_DataType::TensorProto_DataType_FLOAT16:
        return ov::element::f16;
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
        return ov::element::f32;
    case TensorProto_DataType::TensorProto_DataType_INT8:
        return ov::element::i8;
    case TensorProto_DataType::TensorProto_DataType_INT16:
        return ov::element::i16;
    case TensorProto_DataType::TensorProto_DataType_INT32:
        return ov::element::i32;
    case TensorProto_DataType::TensorProto_DataType_INT64:
        return ov::element::i64;
    case TensorProto_DataType::TensorProto_DataType_UINT8:
        return ov::element::u8;
    case TensorProto_DataType::TensorProto_DataType_UINT16:
        return ov::element::u16;
    case TensorProto_DataType::TensorProto_DataType_UINT32:
        return ov::element::u32;
    case TensorProto_DataType::TensorProto_DataType_UINT64:
        return ov::element::u64;
    case TensorProto_DataType::TensorProto_DataType_UNDEFINED:
        return ov::element::dynamic;
    case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
        return ov::element::bf16;
    }
    OPENVINO_THROW("unsupported element type");
}

std::shared_ptr<ov::Node> get_monotonic_range_along_node_rank(const ov::Output<ov::Node>& value,
                                                              int64_t start_value,
                                                              int64_t step) {
    if (value.get_partial_shape().rank().is_static()) {
        const auto range_value =
            get_monotonic_range<int64_t>(value.get_partial_shape().rank().get_length(), start_value, step);
        return v0::Constant::create(ov::element::i64, {range_value.size()}, range_value);
    }

    const auto value_shape = std::make_shared<v0::ShapeOf>(value);
    return std::make_shared<v4::Range>(v0::Constant::create(ov::element::i64, {}, {start_value}),
                                       std::make_shared<v0::ShapeOf>(value_shape),
                                       v0::Constant::create(ov::element::i64, {}, {step}),
                                       ov::element::i64);
}

void validate_scalar_input(const char* input_name,
                           const std::shared_ptr<ov::Node> input,
                           const std::set<ov::element::Type> allowed_types) {
    const auto validated_input_shape = input->get_output_partial_shape(0);
    const auto validated_input_rank = validated_input_shape.rank();

    FRONT_END_GENERAL_CHECK(validated_input_rank.same_scheme({0}) ||
                                (validated_input_rank.same_scheme({1}) && validated_input_shape[0].get_length() == 1),
                            input_name,
                            " needs to be a scalar or 1D, single-element tensor.");

    if (!allowed_types.empty()) {
        const bool data_type_ok = allowed_types.count(input->get_element_type());
        FRONT_END_GENERAL_CHECK(data_type_ok,
                                "Incorrect data type of the ",
                                input_name,
                                " input: ",
                                input->get_element_type());
    }
}

template <typename T>
ov::OutputVector handle_opset6_binary_op(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> lhs_node = node.get_ov_inputs().at(0);
    ov::Output<ov::Node> rhs_node = node.get_ov_inputs().at(1);
    const bool broadcast = node.get_attribute_value<std::int64_t>("broadcast", 0);
    if (broadcast) {
        if (node.has_attribute("axis")) {
            FRONT_END_GENERAL_CHECK(
                lhs_node.get_partial_shape().rank().is_static() && rhs_node.get_partial_shape().rank().is_static(),
                "Input's rank has to be static.");
            auto axis = node.get_attribute_value<std::int64_t>("axis");
            auto lhs_rank = lhs_node.get_partial_shape().rank().get_length();
            auto rhs_rank = rhs_node.get_partial_shape().rank().get_length();
            if (axis < 0)
                axis += lhs_rank;
            if (lhs_rank > axis + rhs_rank) {
                auto ones = v0::Constant::create(ov::element::i64,
                                                 ov::Shape{static_cast<size_t>(lhs_rank - axis - rhs_rank)},
                                                 std::vector<int64_t>(lhs_rank - axis - rhs_rank, 1));
                auto rhs_shape = std::make_shared<v0::ShapeOf>(rhs_node);
                auto new_shape = std::make_shared<v0::Concat>(ov::OutputVector{rhs_shape, ones}, 0);
                rhs_node = std::make_shared<v1::Reshape>(rhs_node, new_shape, false);
            }
        } else {
            rhs_node = std::make_shared<v3::Broadcast>(rhs_node, std::make_shared<v0::ShapeOf>(lhs_node));
        }
    }
    return {std::make_shared<T>(lhs_node, rhs_node)};
}

template ov::OutputVector handle_opset6_binary_op<v1::Add>(const Node& node);
template ov::OutputVector handle_opset6_binary_op<v1::Divide>(const Node& node);
template ov::OutputVector handle_opset6_binary_op<v1::Multiply>(const Node& node);
template ov::OutputVector handle_opset6_binary_op<v1::Subtract>(const Node& node);
template ov::OutputVector handle_opset6_binary_op<v1::LogicalAnd>(const Node& node);

const std::string FAILSAFE_NODE = "ONNX_FAILSAFE_NODE";

std::shared_ptr<v0::Constant> make_failsafe_constant(const ov::element::Type& dtype) {
    const auto failsafe_constant = v0::Constant::create(dtype, ov::Shape{}, {0});
    auto& rt_info = failsafe_constant->get_rt_info();
    rt_info[FAILSAFE_NODE] = true;
    return failsafe_constant;
}

bool is_failsafe_node(const std::shared_ptr<ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(FAILSAFE_NODE) != rt_info.end();
}

const std::string OPTIMIZED_OUT_NODE = "OPTIMIZED_OUT_NODE";

void mark_as_optimized_out(ov::Output<ov::Node>& node_output) {
    node_output.get_rt_info()[OPTIMIZED_OUT_NODE] = true;
}

bool is_optimized_out(const ov::Output<ov::Node>& node_output) {
    const auto& rt_info = node_output.get_rt_info();
    return rt_info.find(OPTIMIZED_OUT_NODE) != rt_info.end();
}

std::string collect_translation_exceptions(const std::shared_ptr<ov::Model>& partially_converted) {
    std::string fully_unsupported_ops = "OpenVINO does not support the following ONNX operations: ";
    std::string additional_error_message = "Errors during ONNX translation: \n";
    const std::string sep = ", ";

    bool unsupported_found = false;
    bool additional_error_found = false;
    for (const auto& op : partially_converted->get_ops()) {
        if (const auto unsupported = std::dynamic_pointer_cast<ov::frontend::onnx::NotSupportedONNXNode>(op)) {
            if (unsupported->additional_error_message().empty()) {
                fully_unsupported_ops += (unsupported->get_attrs().get_opset_name().empty()
                                              ? ""
                                              : unsupported->get_attrs().get_opset_name() + ".") +
                                         unsupported->get_attrs().get_type_name() + sep;
                unsupported_found = true;
            } else {
                additional_error_message += unsupported->additional_error_message();
                additional_error_found = true;
            }
        }
    }
    fully_unsupported_ops = fully_unsupported_ops.substr(0, fully_unsupported_ops.size() - sep.size());
    // remove redundant new line
    additional_error_message =
        (additional_error_message.empty() || additional_error_message[additional_error_message.length() - 1] != '\n')
            ? additional_error_message
            : additional_error_message.erase(additional_error_message.length() - 1);
    if (unsupported_found && additional_error_found) {
        return fully_unsupported_ops + "\n" + additional_error_message;
    } else if (unsupported_found) {
        return fully_unsupported_ops;
    } else if (additional_error_found) {
        return additional_error_message;
    } else {
        return "";
    }
}

}  // namespace  common
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
