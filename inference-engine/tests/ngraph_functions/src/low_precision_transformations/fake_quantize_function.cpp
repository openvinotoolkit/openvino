// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/multiply_add.hpp"
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.lowValues, fakeQuantizeOnData.highValues, fakeQuantizeOnData.lowValues, fakeQuantizeOnData.highValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const float minValue = ngraph::pass::low_precision::DataPrecision::getMinValue(
        params.precisionsOnActivations[0],
        fakeQuantizeOnData.quantizationLevel);

    const float maxValue = ngraph::pass::low_precision::DataPrecision::getMaxValue(params.precisionsOnActivations[0]);

    auto fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        { minValue }, { maxValue }, { minValue }, { maxValue }));

    fakeQuantize = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::FakeQuantize>>(
        *fakeQuantize,
        params.precisionsOnActivations[0]);

    // copy_runtime_info(layer, replacement);
    // replace_node(layer, replacement);

    // fakeQuantize->set_friendly_name("fakeQuantize");
    // fakeQuantize->set_output_type(0, expectedPrecision, inputShape);
    // setOutDataPrecision(fakeQuantize, expectedPrecision);

    const auto convert = std::make_shared<ngraph::opset1::Convert>(fakeQuantize, precision);

    // TODO: MultiplyAdd constant shape is hardcoded
    auto dequantize = std::make_shared<ngraph::op::MultiplyAdd>(
        convert,
        ngraph::opset1::Constant::create(precision, ngraph::Shape{ }, { 1.f }),
        ngraph::opset1::Constant::create(precision, ngraph::Shape{ }, { 0.f }));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
