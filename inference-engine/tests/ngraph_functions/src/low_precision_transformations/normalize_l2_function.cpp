// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/normalize_l2_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> NormalizeL2Function::getOriginal(
    const ngraph::element::Type precision,
    const const std::pair<ngraph::Shape, ngraph::Shape>& shapes,
    const ngraph::element::Type precisionOnActivation,
    const bool fuseMultiply,
    const bool shift) {
    const float low = precisionOnActivation == ngraph::element::u8 ? (0.f + (shift ? 10.f : 0.f)) : (-128.f + (shift ? 10.f : 0.f));
    const float high = precisionOnActivation == ngraph::element::u8 ? 255.f : 127.f;
    const float k = 10.f;

    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(precision, shapes.first);
    paramNode->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        paramNode->output(0), precision, 256, shapes.second,
        { low / k }, { high / k }, { low / k }, { high / k });

    //const ngraph::Shape constShapes({ 1, 1, 1024 });
    //const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
    //    paramNode->output(0), precision, 256, constShapes,
    //    { low / k }, { high / k }, { low / k }, { high / k });

    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 1 }, std::vector<int64_t>{ 1ul });
    axes->set_friendly_name("axes");
    const auto normalizeL2 = std::make_shared<ngraph::opset1::NormalizeL2>(fakeQuantize->output(0), axes, 1e-6, ngraph::op::EpsMode::ADD);
    normalizeL2->set_friendly_name("normalizeL2");

    ngraph::ResultVector results;
    if (fuseMultiply) {
        const auto multiplyConst = std::make_shared<ngraph::op::Constant>(
            precision, ngraph::Shape{ shapes.first[0], shapes.first[1], 1ul, 1ul }, std::vector<float>{ 2.f });
        multiplyConst->set_friendly_name("multiplyConst");
        const auto multiply = std::make_shared<ngraph::opset1::Multiply>(normalizeL2->output(0), multiplyConst);
        multiply->set_friendly_name("multiply");

        results = { std::make_shared<ngraph::opset1::Result>(multiply) };
    } else {
        results = { std::make_shared<ngraph::opset1::Result>(normalizeL2) };
    }

    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ paramNode }, "NormalizeTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> NormalizeL2Function::getReference(
    const ngraph::element::Type precision,
    const std::pair<ngraph::Shape, ngraph::Shape>& shapes,
    const ngraph::element::Type precisionOnActivation,
    const bool fuseMultiply,
    const bool shift) {
    return nullptr;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
