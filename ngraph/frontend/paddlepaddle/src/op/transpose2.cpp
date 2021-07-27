// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs transpose2(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto perm = node.get_attribute<std::vector<int>>("axis");

                    auto rank =
                        static_cast<unsigned long>(data.get_partial_shape().rank().get_length());

                    PDPD_OP_VALIDATION_CHECK(node,
                                             perm.size() == rank,
                                             "transpose2: axis size must equal to data rank!");

                    auto input_order =
                        ngraph::opset6::Constant::create(ngraph::element::i64, {rank}, perm);
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Transpose>(data, input_order)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
