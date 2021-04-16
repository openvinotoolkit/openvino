// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx/onnx_pb.h> // onnx types

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "onnx_common/utils.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector constant_fill(const Node& node)
                {
                    Output<ngraph::Node> target_shape;
                    const auto fill_value = node.get_attribute_value<float>("value", 0.f);
                    const auto dtype = node.get_attribute_value<int64_t>(
                        "dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
                    const auto ng_type = onnx_common::onnx_to_ng_data_type(
                        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(dtype));
                    const auto const_val_to_fill =
                        default_opset::Constant::create(ng_type, {}, {fill_value});
                    const auto input_as_shape =
                        node.get_attribute_value<int64_t>("input_as_shape", 1);
                    if (input_as_shape == 1) // use the first input as target shape
                    {
                        CHECK_VALID_NODE(
                            node,
                            node.get_ng_inputs().size() > 0,
                            "The input which determines output shape was not provided");
                        target_shape = node.get_ng_inputs().at(0);
                        if (node.has_attribute("extra_shape"))
                        {
                            const auto extra_shape =
                                node.get_attribute_value<std::vector<int64_t>>("extra_shape");
                            const auto extra_shape_const = default_opset::Constant::create(
                                target_shape.get_element_type(), {extra_shape.size()}, extra_shape);
                            target_shape = std::make_shared<default_opset::Concat>(
                                OutputVector{target_shape, extra_shape_const}, 0);
                        }
                    }
                    else // use shape attribute as target shape
                    {
                        const auto shape = node.get_attribute_value<std::vector<int64_t>>("shape");
                        target_shape =
                            default_opset::Constant::create(ng_type, {shape.size()}, shape);
                    }

                    return {std::make_shared<default_opset::Broadcast>(const_val_to_fill,
                                                                       target_shape)};
                }

            } // namespace set_1
        }     // namespace op
    }         // namespace onnx_import

} // namespace ngraph
