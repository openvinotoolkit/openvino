// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/normalize_l2_function.hpp"

#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> NormalizeL2Function::getOriginal(
    const ngraph::element::Type precision,
    const std::pair<ngraph::Shape, ngraph::Shape>& shapes,
    const ngraph::element::Type precisionOnActivation,
    const std::vector<uint64_t>& axes,
    const bool fuseMultiply,
    const bool shift) {
    const float low = precisionOnActivation == ngraph::element::u8 ? (0.f + (shift ? 10.f : 0.f)) : (-128.f + (shift ? 10.f : 0.f));
    const float high = precisionOnActivation == ngraph::element::u8 ? 255.f : 127.f;
    const float inputScale = 10.f;
    const float outputScale = 20.f;


    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(precision, shapes.first);
    paramNode->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        paramNode->output(0), precision, 256, shapes.second,
        { low / inputScale }, { high / inputScale }, { low / outputScale }, { high / outputScale });

    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::u64, ngraph::Shape{ axes.size() }, axes);
    axesNode->set_friendly_name("axes");
    const auto normalizeL2 = std::make_shared<ngraph::opset1::NormalizeL2>(fakeQuantize->output(0), axesNode, 1e-6, ngraph::op::EpsMode::ADD);
    normalizeL2->set_friendly_name("normalizeL2");

    ngraph::ResultVector results;
    if (fuseMultiply) {
        const auto multiplyConst = std::make_shared<ngraph::op::Constant>(
            precision, ngraph::Shape{ shapes.first[0], shapes.first[1], 1ul, 1ul }, std::vector<float>{ 2.f });
        multiplyConst->set_friendly_name("multiplyConst");
        const auto multiply = std::make_shared<ngraph::opset1::Multiply>(normalizeL2->output(0), multiplyConst);
        multiply->set_friendly_name("output");

        results = { std::make_shared<ngraph::opset1::Result>(multiply) };
    } else {
        normalizeL2->set_friendly_name("output");
        results = { std::make_shared<ngraph::opset1::Result>(normalizeL2) };
    }

    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ paramNode }, "NormalizeL2Transformation");
    return function;
}

std::shared_ptr<ngraph::Function> NormalizeL2Function::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& shape,
    const ngraph::op::EpsMode& epsMode,
    const std::vector<size_t>& axes,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {

    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision.is_real() ? precision : inputPrecision, shape);

    auto deqStructure = dequantization;
    deqStructure.multiply.outPrecision = precision;
    const auto deq = makeDequantization(input, deqStructure);

    const auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ axes.size() }, axes);
    const auto normalizeL2 = std::make_shared<ngraph::opset1::NormalizeL2>(deq, axesNode, 1e-6, epsMode);
    normalizeL2->set_friendly_name("output");
    auto& rtInfo = normalizeL2->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("normalizeL2");

    ngraph::ResultVector results = { std::make_shared<ngraph::opset1::Result>(normalizeL2) };
    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "NormalizeL2Transformation");
    return function;
}

std::shared_ptr<ngraph::Function> NormalizeL2Function::getReference(
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& shape,
    const ngraph::op::EpsMode& epsMode,
    const std::vector<size_t>& axes,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision.is_real() ? precision : inputPrecision, shape);

    auto deqBeforeStructure = dequantizationBefore;
    if (dequantizationAfter.empty()) {
        deqBeforeStructure.multiply.outPrecision = precision;
    }

    const auto deqBefore = makeDequantization(input, deqBeforeStructure);

    const auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ axes.size() }, axes);
    const auto normalizeL2 = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::NormalizeL2>>(
        std::vector<ngraph::element::Type>{ element::f32, element::f32 },
        std::vector<ngraph::element::Type>{dequantizationAfter.empty() ? precision : element::f32},
        ngraph::op::TemporaryReplaceOutputType(deqBefore, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(axesNode, element::f32).get(),
        1e-6,
        epsMode);
    auto& rtInfo = normalizeL2->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("normalizeL2");

    auto deqAfterStructure = dequantizationAfter;
    deqAfterStructure.multiply.outPrecision = precision;
    const auto deqAfter = makeDequantization(normalizeL2, deqAfterStructure);

    deqAfter->set_friendly_name("output");

    ngraph::ResultVector results = { std::make_shared<ngraph::opset1::Result>(deqAfter) };
    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "NormalizeL2Transformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
