// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fill_constant_batch_size_like.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs fill_constant_batch_size_like(const NodeContext& node)
                {
                    // TODO to Support other data types other than FP32 #55263
                    auto input_dim_idx = node.get_attribute<int32_t>("input_dim_idx", 0);
                    auto output_dim_idx = node.get_attribute<int32_t>("output_dim_idx", 0);
                    auto value = node.get_attribute<float>("value");
                    auto shapes = node.get_attribute<std::vector<int32_t>>("shape");
                    auto input = node.get_ng_input("Input");
                    auto partial_shape = input.get_partial_shape();
                    PDPD_OP_VALIDATION_CHECK(
                        node,
                        partial_shape.is_static(),
                        "fill_constant_batch_size_like: must use static shape.");
                    auto static_shape = partial_shape.get_shape();
                    PDPD_OP_VALIDATION_CHECK(node,
                                             input_dim_idx < (int32_t)static_shape.size(),
                                             "fill_constant_batch_size_like: input_dim_idx "
                                             "should not exceed input dims.");
                    PDPD_OP_VALIDATION_CHECK(node,
                                             "fill_constant_batch_size_like: output_dim_idx "
                                             "should not exceed shapes dims.");
                    shapes[output_dim_idx] = static_shape[input_dim_idx];
                    auto dtype = node.get_attribute<element::Type>("dtype");
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Constant>(
                            dtype, Shape(shapes.begin(), shapes.end()), value)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph