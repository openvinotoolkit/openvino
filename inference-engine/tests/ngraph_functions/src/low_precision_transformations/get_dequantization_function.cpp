// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/get_dequantization_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include <low_precision/common/fake_quantize_dequantization.hpp>


namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ngraph::Function> GetDequantizationFunction::getOriginal(
    bool isConvert, bool isSubtract, size_t subDataInput, size_t mulDataInput) {
    const std::shared_ptr<ngraph::Node> input = std::make_shared<ngraph::opset1::Parameter>(
        ngraph::element::f32,
        ngraph::Shape{ 1, 3, 10, 10 });

    const auto convert = isConvert ? std::make_shared<ngraph::opset1::Convert>(input, ngraph::element::f32) : nullptr;
    std::shared_ptr<ngraph::Node> parent = isConvert ? convert : input;

    auto subConst = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, Shape{}, 1.f);
    const auto& subArg0 = subDataInput == 0 ? parent : subConst;
    const auto& subArg1 = subDataInput == 0 ? subConst : parent;
    const auto subtract = isSubtract ? std::make_shared<ngraph::opset1::Subtract>(subArg0, subArg1) : nullptr;

    if (subtract != nullptr) {
        parent = subtract;
    }

    auto mulConst = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, Shape{}, 1.f);
    const auto& mulArg0 = mulDataInput == 0 ? parent : mulConst;
    const auto& mulArg1 = mulDataInput == 0 ? mulConst : parent;
    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(mulArg0, mulArg1);

    return std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ std::make_shared<ngraph::opset1::Result>(multiply) },
        ngraph::ParameterVector{ as_type_ptr<op::v0::Parameter>(input) },
        "Dequantization");
}

std::shared_ptr<ngraph::Function> GetDequantizationFunction::getReference(
    ngraph::pass::low_precision::FakeQuantizeDequantization dequantization) {
    return std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ std::make_shared<ngraph::opset1::Result>(dequantization.multiply) },
        ngraph::ParameterVector{ as_type_ptr<op::v0::Parameter>(dequantization.data.get_node_shared_ptr()) },
        "Dequantization");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
