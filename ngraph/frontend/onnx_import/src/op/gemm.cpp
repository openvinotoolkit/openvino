// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/matmul.hpp"
#include "ngraph/op/multiply.hpp"
#include "op/gemm.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector gemm(const Node& node)
                {
                    OutputVector inputs{node.get_ng_inputs()};
                    Output<ngraph::Node> input_a = inputs.at(0);
                    Output<ngraph::Node> input_b = inputs.at(1);
                    Output<ngraph::Node> input_c;

                    if (inputs.size() == 3)
                    {
                        input_c = inputs.at(2);
                    }
                    else
                    {
                        input_c = default_opset::Constant::create(
                            input_b.get_element_type(), ngraph::Shape{}, {0});
                    }

                    const auto alpha = node.get_attribute_value<float>("alpha", 1);
                    const auto beta = node.get_attribute_value<float>("beta", 1);

                    const auto alpha_node = default_opset::Constant::create(
                        input_b.get_element_type(), Shape{}, {alpha});
                    const auto beta_node = default_opset::Constant::create(
                        input_c.get_element_type(), Shape{}, {beta});

                    const bool trans_a = node.get_attribute_value<int64_t>("transA", 0);
                    const bool trans_b = node.get_attribute_value<int64_t>("transB", 0);

                    if (trans_a)
                    {
                        input_a = ngraph::builder::opset1::transpose(input_a);
                    }

                    if (trans_b)
                    {
                        input_b = ngraph::builder::opset1::transpose(input_b);
                    }

                    input_a = ngraph::builder::opset1::flatten(input_a, 1);
                    input_b = ngraph::builder::opset1::flatten(input_b, 1);

                    std::shared_ptr<ngraph::Node> matmul_node =
                        std::make_shared<default_opset::MatMul>(input_a, input_b);

                    if (alpha != 1)
                    {
                        matmul_node =
                            std::make_shared<default_opset::Multiply>(matmul_node, alpha_node);
                    }

                    auto beta_times_input_c =
                        std::make_shared<default_opset::Multiply>(beta_node, input_c);

                    return OutputVector{
                        std::make_shared<default_opset::Add>(matmul_node, beta_times_input_c)};
                }

            } // namespace set_1

            namespace set_6
            {
                OutputVector gemm(const Node& node)
                {
                    OutputVector inputs{node.get_ng_inputs()};
                    Output<ngraph::Node> input_a = inputs.at(0);
                    Output<ngraph::Node> input_b = inputs.at(1);
                    Output<ngraph::Node> input_c;

                    if (inputs.size() == 3)
                    {
                        input_c = inputs.at(2);
                    }
                    else
                    {
                        input_c = default_opset::Constant::create(
                            input_b.get_element_type(), ngraph::Shape{}, {0});
                    }

                    const auto alpha = node.get_attribute_value<float>("alpha", 1);
                    const auto beta = node.get_attribute_value<float>("beta", 1);

                    const auto alpha_node = default_opset::Constant::create(
                        input_b.get_element_type(), Shape{}, {alpha});
                    const auto beta_node = default_opset::Constant::create(
                        input_c.get_element_type(), Shape{}, {beta});

                    const bool trans_a = node.get_attribute_value<int64_t>("transA", 0);
                    const bool trans_b = node.get_attribute_value<int64_t>("transB", 0);

                    std::shared_ptr<ngraph::Node> matmul_node =
                        std::make_shared<default_opset::MatMul>(input_a, input_b, trans_a, trans_b);

                    if (alpha != 1)
                    {
                        matmul_node =
                            std::make_shared<default_opset::Multiply>(matmul_node, alpha_node);
                    }

                    auto beta_times_input_c =
                        std::make_shared<default_opset::Multiply>(beta_node, input_c);

                    return OutputVector{
                        std::make_shared<default_opset::Add>(matmul_node, beta_times_input_c)};
                }

            } // namespace set_6

        } // namespace op

    } // namespace  onnx_import

} // namespace  ngraph
