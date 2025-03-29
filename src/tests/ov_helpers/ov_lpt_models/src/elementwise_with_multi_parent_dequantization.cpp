// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/elementwise_with_multi_parent_dequantization.hpp"
#include "low_precision/network_helper.hpp"

#include "openvino/opsets/opset1.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> ElementwiseWithMultiParentDequantizationFunction::get(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const ov::pass::low_precision::LayerTransformation::Params& params,
    const ov::element::Type& precision1,
    const ov::builder::subgraph::DequantizationOperations& dequantization1,
    const ov::element::Type& precision2,
    const ov::builder::subgraph::DequantizationOperations& dequantization2) {
    const auto input1_1 = std::make_shared<ov::opset1::Parameter>(precision1, inputShape);
    const auto input1_2 = std::make_shared<ov::opset1::Parameter>(precision1, ov::Shape({ inputShape[0], inputShape[1], 1, 1 }));
    const std::shared_ptr<ov::Node> multiply1 = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        ov::opset1::Multiply(ov::op::TemporaryReplaceOutputType(input1_1, ov::element::f32).get(),
                             ov::op::TemporaryReplaceOutputType(input1_2, ov::element::f32).get()),
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{});

    const std::shared_ptr<ov::Node> parent1 = dequantization1.empty() ? multiply1 : makeDequantization(multiply1, dequantization1);

    const auto input2_1 = std::make_shared<ov::opset1::Parameter>(precision1, inputShape);
    const auto input2_2 = std::make_shared<ov::opset1::Parameter>(precision1, ov::Shape({ inputShape[0], inputShape[1], 1, 1 }));
    const std::shared_ptr<ov::Node> multiply2 = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        ov::opset1::Multiply(ov::op::TemporaryReplaceOutputType(input2_1, ov::element::f32).get(),
                             ov::op::TemporaryReplaceOutputType(input2_2, ov::element::f32).get()),
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{});

    const std::shared_ptr<ov::Node> parent2 = dequantization2.empty() ? multiply2 : makeDequantization(multiply2, dequantization2);

    const auto add = std::make_shared<ov::opset1::Add>(parent1, parent2);
    add->set_friendly_name("output");
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = "add";

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(add) };
    ov::ParameterVector parameters = { input1_1, input1_2, input2_1, input2_2 };
    return std::make_shared<ov::Model>(results, parameters, "ElementwiseWithMultiParentDequantization");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
