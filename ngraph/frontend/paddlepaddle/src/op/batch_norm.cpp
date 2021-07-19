// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_norm.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs batch_norm(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto gamma = node.get_ng_input("Scale");
                    auto beta = node.get_ng_input("Bias");
                    auto mean = node.get_ng_input("Mean");
                    auto variance = node.get_ng_input("Variance");
                    auto data_layout = node.get_attribute<std::string>("data_layout");

                    PDPD_ASSERT((data_layout == "NCHW" || data_layout == "NHWC"),
                                "Not supported input data layout!");
                    if (data_layout == "NCHW")
                    {
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::BatchNormInference>(
                                data,
                                gamma,
                                beta,
                                mean,
                                variance,
                                node.get_attribute<float>("epsilon"))},
                            {"Y"});
                    }
                    else
                    {
                        auto input_order = ngraph::opset6::Constant::create(
                            ngraph::element::i64, {4}, {0, 3, 1, 2});
                        auto data_nchw =
                            std::make_shared<ngraph::opset6::Transpose>(data, input_order);
                        auto node_batch_norm = std::make_shared<ngraph::opset6::BatchNormInference>(
                            data_nchw,
                            gamma,
                            beta,
                            mean,
                            variance,
                            node.get_attribute<float>("epsilon"));
                        auto output_order = ngraph::opset6::Constant::create(
                            ngraph::element::i64, {4}, {0, 2, 3, 1});
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Transpose>(node_batch_norm,
                                                                         output_order)},
                            {"Y"});
                    }
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
