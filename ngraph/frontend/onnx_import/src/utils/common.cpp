// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx/onnx_pb.h> // onnx types

#include "default_opset.hpp"
#include "ngraph/graph_util.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace common
        {
            const ngraph::element::Type& get_ngraph_element_type(int64_t onnx_type)
            {
                switch (onnx_type)
                {
                case ONNX_NAMESPACE::TensorProto_DataType_BOOL: return element::boolean;
                case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: return element::f64;
                case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: return element::f16;
                case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: return element::f32;
                case ONNX_NAMESPACE::TensorProto_DataType_INT8: return element::i8;
                case ONNX_NAMESPACE::TensorProto_DataType_INT16: return element::i16;
                case ONNX_NAMESPACE::TensorProto_DataType_INT32: return element::i32;
                case ONNX_NAMESPACE::TensorProto_DataType_INT64: return element::i64;
                case ONNX_NAMESPACE::TensorProto_DataType_UINT8: return element::u8;
                case ONNX_NAMESPACE::TensorProto_DataType_UINT16: return element::u16;
                case ONNX_NAMESPACE::TensorProto_DataType_UINT32: return element::u32;
                case ONNX_NAMESPACE::TensorProto_DataType_UINT64: return element::u64;
                case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: return element::dynamic;
                case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: return element::bf16;
                }
#ifdef NGRAPH_USE_PROTOBUF_LITE
                throw ngraph_error("unsupported element type");
#else
                throw ngraph_error(
                    "unsupported element type: " +
                    ONNX_NAMESPACE::TensorProto_DataType_Name(
                        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(onnx_type)));
#endif
            }

            std::shared_ptr<ngraph::Node> get_monotonic_range_along_node_rank(
                const Output<ngraph::Node>& value, int64_t start_value, int64_t step)
            {
                if (value.get_partial_shape().rank().is_static())
                {
                    const auto range_value = get_monotonic_range<int64_t>(
                        value.get_partial_shape().rank().get_length(), start_value, step);
                    return default_opset::Constant::create(
                        element::i64, {range_value.size()}, range_value);
                }

                const auto value_shape = std::make_shared<default_opset::ShapeOf>(value);
                return std::make_shared<default_opset::Range>(
                    default_opset::Constant::create(element::i64, {}, {start_value}),
                    std::make_shared<default_opset::ShapeOf>(value_shape),
                    default_opset::Constant::create(element::i64, {}, {step}),
                    element::i64);
            }

            void validate_scalar_input(const char* input_name,
                                       const std::shared_ptr<ngraph::Node> input,
                                       const std::set<element::Type> allowed_types)
            {
                const auto validated_input_shape = input->get_output_partial_shape(0);
                const auto validated_input_rank = validated_input_shape.rank();

                NGRAPH_CHECK(validated_input_rank.same_scheme({0}) ||
                                 (validated_input_rank.same_scheme({1}) &&
                                  validated_input_shape[0].get_length() == 1),
                             input_name,
                             " needs to be a scalar or 1D, single-element tensor.");

                if (!allowed_types.empty())
                {
                    const bool data_type_ok = allowed_types.count(input->get_element_type());
                    NGRAPH_CHECK(data_type_ok,
                                 "Incorrect data type of the ",
                                 input_name,
                                 " input: ",
                                 input->get_element_type());
                }
            }

            template <typename T>
            OutputVector handle_opset6_binary_op(const Node& node)
            {
                const Output<ngraph::Node> lhs_node = node.get_ng_inputs().at(0);
                Output<ngraph::Node> rhs_node = node.get_ng_inputs().at(1);
                const bool broadcast = node.get_attribute_value<std::int64_t>("broadcast", 0);
                if (broadcast)
                {
                    if (node.has_attribute("axis"))
                    {
                        NGRAPH_CHECK(lhs_node.get_partial_shape().rank().is_static() &&
                                         rhs_node.get_partial_shape().rank().is_static(),
                                     "Input's rank has to be static.");
                        auto axis = node.get_attribute_value<std::int64_t>("axis");
                        auto lhs_rank = lhs_node.get_partial_shape().rank().get_length();
                        auto rhs_rank = rhs_node.get_partial_shape().rank().get_length();
                        if (axis < 0)
                            axis += lhs_rank;
                        if (lhs_rank > axis + rhs_rank)
                        {
                            auto ones = default_opset::Constant::create(
                                element::i64,
                                Shape{static_cast<size_t>(lhs_rank - axis - rhs_rank)},
                                std::vector<int64_t>(lhs_rank - axis - rhs_rank, 1));
                            auto rhs_shape = std::make_shared<default_opset::ShapeOf>(rhs_node);
                            auto new_shape = std::make_shared<default_opset::Concat>(
                                OutputVector{rhs_shape, ones}, 0);
                            rhs_node = std::make_shared<default_opset::Reshape>(
                                rhs_node, new_shape, false);
                        }
                    }
                    else
                    {
                        rhs_node = std::make_shared<default_opset::Broadcast>(
                            rhs_node, std::make_shared<default_opset::ShapeOf>(lhs_node));
                    }
                }
                return {std::make_shared<T>(lhs_node, rhs_node)};
            }

            template OutputVector handle_opset6_binary_op<default_opset::Add>(const Node& node);
            template OutputVector handle_opset6_binary_op<default_opset::Divide>(const Node& node);
            template OutputVector
                handle_opset6_binary_op<default_opset::Multiply>(const Node& node);
            template OutputVector
                handle_opset6_binary_op<default_opset::Subtract>(const Node& node);

        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
