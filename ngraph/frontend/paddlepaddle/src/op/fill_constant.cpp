// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fill_constant.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs fill_constant(const NodeContext& node)
                {
                    auto shape = node.get_attribute<std::vector<int64_t>>("shape");
                    auto dtype = node.get_attribute<ngraph::element::Type>("dtype");
                    Output<Node> value_node;
                    Output<Node> shape_node;
                    if (node.has_ng_input("ValueTensor"))
                    {
                        value_node = node.get_ng_input("ValueTensor");
                    }
                    else if (dtype == element::i32)
                    {
                        int32_t value = static_cast<int32_t>(node.get_attribute<float>("value"));
                        value_node = opset6::Constant::create(dtype, {1}, {value});
                    }
                    else if (dtype == element::f32)
                    {
                        float value = node.get_attribute<float>("value");
                        value_node = opset6::Constant::create(dtype, {1}, {value});
                    }
                    else if (dtype == element::i64)
                    {
                        int64_t value = static_cast<int64_t>(node.get_attribute<float>("value"));
                        value_node = opset6::Constant::create(dtype, {1}, {value});
                    }
                    else
                    {
                        PDPD_ASSERT(false, "fill_constant only supports i32, f32, i64");
                    }

                    PDPD_ASSERT(shape.size() > 0 || node.has_ng_input("ShapeTensor") ||
                                    node.has_ng_input("ShapeTensorList"),
                                "fill_constant shape not set");

                    if (node.has_ng_input("ShapeTensor"))
                    {
                        shape_node = node.get_ng_input("ShapeTensor");
                    }
                    else if (node.has_ng_input("ShapeTensorList"))
                    {
                        auto shape_tensor_list = node.get_ng_inputs("ShapeTensorList");
                        shape_node =
                            Output<Node>{std::make_shared<opset6::Concat>(shape_tensor_list, 0)};
                    }
                    else
                    {
                        shape_node = opset6::Constant::create(element::i64, {shape.size()}, shape);
                    }

                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Broadcast>(value_node, shape_node)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
