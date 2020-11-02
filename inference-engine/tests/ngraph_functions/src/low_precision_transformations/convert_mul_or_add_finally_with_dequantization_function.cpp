// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/convert_mul_or_add_finally_with_dequantization_function.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "low_precision/network_helper.hpp"
#include <legacy/ngraph_ops/scaleshift.hpp>
#include "low_precision/common/dequantization_op.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> ConvertMulOrAddWithDequantizationFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision,
    const std::vector<float>& multiplyConst) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    const auto reluOriginal = ngraph::opset1::Relu(
        ngraph::op::TemporaryReplaceOutputType(input, element::f32).get());

    std::shared_ptr<ngraph::opset1::Relu> relu = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Relu>>(
        reluOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});


    const auto multiply = std::make_shared<ngraph::pass::low_precision::DequantizationMultiply>(relu,
                                                            std::make_shared<opset1::Constant>(element::f32, inputShape, multiplyConst));

    multiply->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input },
                                              "ConvertMulOrAddTransformationWithDequantization");
}

std::shared_ptr<ngraph::Function> ConvertMulOrAddWithDequantizationFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision,
    const std::vector<float>& multiplyConst) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    const auto reluOriginal = ngraph::opset1::Relu(
        ngraph::op::TemporaryReplaceOutputType(input, element::f32).get());

    std::shared_ptr<ngraph::opset1::Relu> relu = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Relu>>(
        reluOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

    const auto weights = std::make_shared<opset1::Constant>(element::f32, inputShape, multiplyConst);
    const auto bias = std::make_shared<opset1::Constant>(element::f32, inputShape, 0.0);
    std::shared_ptr<Node> scaleShift = std::make_shared<ngraph::op::ScaleShiftIE>(relu, weights, bias);

    scaleShift = low_precision::NetworkHelper::markAsDequantizationOp(scaleShift);

    scaleShift->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(scaleShift) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvertMulOrAddTransformationWithDequantization");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
