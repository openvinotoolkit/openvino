// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/normalize_l2.hpp"

#include <ov_ops/type_relaxed.hpp>
#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> NormalizeL2Function::getOriginal(
    const ov::element::Type precision,
    const std::pair<ov::PartialShape, ov::Shape>& shapes,
    const ov::element::Type precisionOnActivation,
    const std::vector<uint64_t>& axes,
    const bool fuseMultiply,
    const bool shift) {
    const float low = precisionOnActivation == ov::element::u8 ? (0.f + (shift ? 10.f : 0.f)) : (-128.f + (shift ? 10.f : 0.f));
    const float high = precisionOnActivation == ov::element::u8 ? 255.f : 127.f;
    const float inputScale = 10.f;
    const float outputScale = 20.f;


    const auto paramNode = std::make_shared<ov::opset1::Parameter>(precision, shapes.first);
    paramNode->set_friendly_name("input");

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
        paramNode->output(0), precision, 256, shapes.second,
        { low / inputScale }, { high / inputScale }, { low / outputScale }, { high / outputScale });

    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ axes.size() }, axes);
    axesNode->set_friendly_name("axes");
    const auto normalizeL2 = std::make_shared<ov::opset1::NormalizeL2>(fakeQuantize->output(0), axesNode, 1e-6, ov::op::EpsMode::ADD);
    normalizeL2->set_friendly_name("normalizeL2");

    ov::ResultVector results;
    if (fuseMultiply) {
        ov::Shape constantShape(4ul, 1ul);
        constantShape[0] = shapes.first[0].get_length();
        constantShape[1] = shapes.first[1].get_length();

        const auto multiplyConst = std::make_shared<ov::op::v0::Constant>(precision, constantShape, std::vector<float>{ 2.f });
        multiplyConst->set_friendly_name("multiplyConst");
        const auto multiply = std::make_shared<ov::opset1::Multiply>(normalizeL2->output(0), multiplyConst);
        multiply->set_friendly_name("output");

        results = { std::make_shared<ov::opset1::Result>(multiply) };
    } else {
        normalizeL2->set_friendly_name("output");
        results = { std::make_shared<ov::opset1::Result>(normalizeL2) };
    }

    const auto function = std::make_shared<ov::Model>(results, ov::ParameterVector{ paramNode }, "NormalizeL2Transformation");
    return function;
}

std::shared_ptr<ov::Model> NormalizeL2Function::getOriginal(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::PartialShape& shape,
    const ov::op::EpsMode& epsMode,
    const std::vector<size_t>& axes,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {

    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, shape);

    auto deqStructure = dequantization;
    deqStructure.multiply.outPrecision = precision;
    const auto deq = makeDequantization(input, deqStructure);

    const auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ axes.size() }, axes);
    const auto normalizeL2 = std::make_shared<ov::opset1::NormalizeL2>(deq, axesNode, 1e-6, epsMode);
    normalizeL2->set_friendly_name("output");
    auto& rtInfo = normalizeL2->get_rt_info();
    rtInfo["Variant::std::string"] = "normalizeL2";

    ov::ResultVector results = { std::make_shared<ov::opset1::Result>(normalizeL2) };
    const auto function = std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "NormalizeL2Transformation");
    return function;
}

std::shared_ptr<ov::Model> NormalizeL2Function::getReference(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::PartialShape& shape,
    const ov::op::EpsMode& epsMode,
    const std::vector<size_t>& axes,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, shape);

    auto deqBeforeStructure = dequantizationBefore;
    if (dequantizationAfter.empty()) {
        deqBeforeStructure.multiply.outPrecision = precision;
    }

    const auto deqBefore = makeDequantization(input, deqBeforeStructure);

    const auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ axes.size() }, axes);
    const auto normalizeL2 = std::make_shared<ov::op::TypeRelaxed<ov::opset1::NormalizeL2>>(
        std::vector<ov::element::Type>{ov::element::f32, axesNode->output(0).get_element_type()},
        std::vector<ov::element::Type>{dequantizationAfter.empty() ? precision : ov::element::f32},
        ov::op::TemporaryReplaceOutputType(deqBefore, ov::element::f32).get(),
        axesNode,
        1e-6,
        epsMode);
    auto& rtInfo = normalizeL2->get_rt_info();
    rtInfo["Variant::std::string"] = "normalizeL2";

    auto deqAfterStructure = dequantizationAfter;
    deqAfterStructure.multiply.outPrecision = precision;
    const auto deqAfter = makeDequantization(normalizeL2, deqAfterStructure);

    deqAfter->set_friendly_name("output");

    ov::ResultVector results = { std::make_shared<ov::opset1::Result>(deqAfter) };
    const auto function = std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "NormalizeL2Transformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
