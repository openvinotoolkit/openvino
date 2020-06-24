// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/concat_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/multiply_add.hpp"
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

// TODO: copy/paster from LayerTransformation::getQuantizationInterval
std::pair<float, float> getQuantizationInterval(const ngraph::element::Type precision) {
    const bool unsignedInterval = precision == ngraph::element::u8;
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

// TODO: use dequantization operations only (remove FQ later)
std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
    const ngraph::element::Type ngPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
    const float low = interval.first;
    const float hight = interval.second;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1, ngPrecision, 256ul, { 1ul },
        { low }, { hight }, { low }, { hight });

    //const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2] / 2, inputShape[3] / 2 };
    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape2));
    input1->set_friendly_name("input2");
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2, ngPrecision, 256ul, { 1ul },
        { low / 2.f }, { hight / 2.f }, { low / 2.f }, { hight / 2.f });

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1, input2 }, "ConcatTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");
    return nullptr;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
