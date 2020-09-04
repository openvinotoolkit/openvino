// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_and_scale_shift_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FuseFakeQuantizeAndScaleShiftFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const std::shared_ptr<Node> fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    const std::shared_ptr<Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        fakeQuantize,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, std::vector<float>({ 150 })));

    const std::shared_ptr<Node> add = std::make_shared<ngraph::opset1::Add>(
        multiply,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, std::vector<float>({ 127.5 })));

    const ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeAndScaleShiftFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
