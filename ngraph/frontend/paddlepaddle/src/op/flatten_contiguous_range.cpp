// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "flatten_contiguous_range.hpp"
#include <ngraph/builder/reshape.hpp>
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs flatten_contiguous_range(const NodeContext& node)
                {
                    auto x_node = node.get_ng_input("X");
                    auto shape_of_x = std::make_shared<opset6::ShapeOf>(x_node);
                    int dims = x_node.get_partial_shape().rank().get_length();
                    auto start_axis = node.get_attribute<int32_t>("start_axis");
                    auto stop_axis = node.get_attribute<int32_t>("stop_axis");

                    auto axis1_begin = opset6::Constant::create(element::i64, {1}, {0});
                    auto axis1_end = opset6::Constant::create(element::i64, {1}, {start_axis});
                    auto axis1 = std::make_shared<opset6::StridedSlice>(shape_of_x,
                                                                        axis1_begin,
                                                                        axis1_end,
                                                                        std::vector<int64_t>{0},
                                                                        std::vector<int64_t>{0});
                    OutputVector axes{axis1,
                                      opset6::Constant::create(element::i64, Shape{1}, {-1.0})};

                    if (stop_axis < dims - 1)
                    {
                        auto axis2_begin =
                            opset6::Constant::create(element::i64, {1}, {stop_axis + 1});
                        auto axis2_end = opset6::Constant::create(element::i64, {1}, {dims});
                        auto axis2_node =
                            std::make_shared<opset6::StridedSlice>(shape_of_x,
                                                                   axis2_begin,
                                                                   axis2_end,
                                                                   std::vector<int64_t>{0},
                                                                   std::vector<int64_t>{0});
                        axes.push_back(axis2_node);
                    }

                    auto new_shape_node = std::make_shared<opset6::Concat>(axes, 0);
                    return node.default_single_output_mapping(
                        {std::make_shared<opset6::Reshape>(x_node, new_shape_node, true)}, {"Out"});
                }
            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
