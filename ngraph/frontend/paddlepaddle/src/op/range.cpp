// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs range(const NodeContext& node)
                {
                    auto start = node.get_ng_input("Start");
                    auto stop = node.get_ng_input("End");
                    auto step = node.get_ng_input("Step");
                    auto type = node.get_out_port_type("Out");
                    PDPD_ASSERT(type == element::i64 || type == element::i32 ||
                                    type == element::f32,
                                "Only supports int32, int64, float32");

                    const auto axis = ngraph::opset6::Constant::create(element::i64, Shape{}, {0});
                    auto start_scalar = std::make_shared<ngraph::opset6::Squeeze>(start, axis);
                    auto stop_scalar = std::make_shared<ngraph::opset6::Squeeze>(stop, axis);
                    auto step_scalar = std::make_shared<ngraph::opset6::Squeeze>(step, axis);

                    // TODO to support other data types other than FP32 #55267
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Range>(
                            start_scalar, stop_scalar, step_scalar, type)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
