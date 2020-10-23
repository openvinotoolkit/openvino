// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/common/dequantization_op.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, element::f32, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool updatePrecisions,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const ngraph::element::Type fakeQuantizeOutputPrecision,
    const std::vector<float>& expectedSubtractValues,
    const std::vector<float>& expectedMultiplyValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input,
        element::f32,
        fakeQuantizeOnData.quantizationLevel,
        fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues,
        fakeQuantizeOnData.inputHighValues,
        fakeQuantizeOnData.outputLowValues,
        fakeQuantizeOnData.outputHighValues));
    std::shared_ptr<Node> parent = fakeQuantize;

    if (updatePrecisions) {
        const std::shared_ptr<ngraph::opset1::Convert> convert = std::make_shared<DequantizationConvert>(parent, element::f32);
        parent = convert;

        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize, fakeQuantizeOutputPrecision);
    } else {
        if (fakeQuantize->get_output_element_type(0) != element::f32) {
            const std::shared_ptr<ngraph::opset1::Convert> convert = std::make_shared<DequantizationConvert>(parent, element::f32);
            parent = convert;
        }
    }

    const std::shared_ptr<ngraph::opset1::Subtract> subtract = expectedSubtractValues.empty() ?
        nullptr :
        std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(
            parent,
            ngraph::opset1::Constant::create(
                element::f32,
                expectedSubtractValues.size() == 1ul ? ngraph::Shape{ } : ngraph::Shape{ expectedSubtractValues.size() },
                expectedSubtractValues),
            ngraph::op::AutoBroadcastSpec::NUMPY);
    if (subtract != nullptr) {
        parent = subtract;
    }

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = expectedMultiplyValues.empty() ?
        nullptr :
        std::make_shared<DequantizationMultiply>(
            parent,
            ngraph::opset1::Constant::create(
                element::f32,
                expectedMultiplyValues.size() == 1ul ? ngraph::Shape{ } : ngraph::Shape{ expectedMultiplyValues.size() },
                expectedMultiplyValues));
    if (multiply != nullptr) {
        parent = multiply;
    }
    parent->set_friendly_name("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
