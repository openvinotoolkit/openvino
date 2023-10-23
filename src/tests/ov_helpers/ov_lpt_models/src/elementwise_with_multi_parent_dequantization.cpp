// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/elementwise_with_multi_parent_dequantization.hpp"
#include "low_precision/network_helper.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_models/builders.hpp"
#include "ov_models/subgraph_builders.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ElementwiseWithMultiParentDequantizationFunction::get(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ov::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ngraph::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2) {
    const auto input1_1 = std::make_shared<ngraph::opset1::Parameter>(precision1, inputShape);
    const auto input1_2 = std::make_shared<ngraph::opset1::Parameter>(precision1, ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }));
    const std::shared_ptr<ngraph::Node> multiply1 = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Multiply>>(
        opset1::Multiply(
            ov::op::TemporaryReplaceOutputType(input1_1, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(input1_2, element::f32).get()),
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{});

    const std::shared_ptr<ngraph::Node> parent1 = dequantization1.empty() ? multiply1 : makeDequantization(multiply1, dequantization1);

    const auto input2_1 = std::make_shared<ngraph::opset1::Parameter>(precision1, inputShape);
    const auto input2_2 = std::make_shared<ngraph::opset1::Parameter>(precision1, ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }));
    const std::shared_ptr<ngraph::Node> multiply2 = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Multiply>>(
        opset1::Multiply(
            ov::op::TemporaryReplaceOutputType(input2_1, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(input2_2, element::f32).get()),
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{});

    const std::shared_ptr<ngraph::Node> parent2 = dequantization2.empty() ? multiply2 : makeDequantization(multiply2, dequantization2);

    const auto add = std::make_shared<ngraph::opset1::Add>(parent1, parent2);
    add->set_friendly_name("output");
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = "add";

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    ngraph::ParameterVector parameters = { input1_1, input1_2, input2_1, input2_2 };
    return std::make_shared<ngraph::Function>(results, parameters, "ElementwiseWithMultiParentDequantization");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
