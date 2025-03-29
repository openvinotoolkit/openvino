// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_lpt_models/variadic_split.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {
    std::shared_ptr<ov::Model> VariadicSplitFunction::getOriginal(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths) {
        const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

        const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
        const auto constantAxis = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{}, splitedAxis);
        const auto constantLengths =
            std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{splitLengths.size()}, splitLengths);
        const std::shared_ptr<Node> variadicSplit = std::make_shared<ov::opset1::VariadicSplit>(dequantizationOp, constantAxis, constantLengths);

        ov::ResultVector results;
        for (size_t i = 0; i < splitLengths.size(); ++i) {
            results.push_back(std::make_shared<ov::opset1::Result>(variadicSplit->output(i)));
        }
        return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "VariadicSplitFunction");
    }

std::shared_ptr<ov::Model> VariadicSplitFunction::getOriginal(
    const ov::element::Type originalFunctionPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    const int64_t splitedAxis,
    const std::vector<size_t>& splitLengths) {
    const auto input = std::make_shared<ov::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fq = fakeQuantize.empty() ? nullptr :
        ov::test::utils::make_fake_quantize(
            input,
            originalFunctionPrecision,
            fakeQuantize.quantizationLevel,
            fakeQuantize.constantShape,
            fakeQuantize.inputLowValues,
            fakeQuantize.inputHighValues,
            fakeQuantize.outputLowValues,
            fakeQuantize.outputHighValues);

    const auto constantAxis = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{}, splitedAxis);
    const auto constantLengths =
        std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{splitLengths.size()}, splitLengths);
    const std::shared_ptr<Node> variadicSplit =
        std::make_shared<ov::opset1::VariadicSplit>(fakeQuantize.empty() ? input : fq, constantAxis, constantLengths);

    ov::ResultVector results;
    for (size_t i = 0; i < splitLengths.size(); ++i) {
        results.push_back(std::make_shared<ov::opset1::Result>(variadicSplit->output(i)));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "VariadicSplitFunction");
}

std::shared_ptr<ov::Model> VariadicSplitFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::element::Type inputPrecision,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const std::vector<ov::builder::subgraph::DequantizationOperations>& dequantizationAfter,
    const int64_t splitedAxis,
    const std::vector<size_t>& splitLengths) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);

    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto constantAxis = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{}, splitedAxis);
    const auto constantLengths =
        std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{splitLengths.size()}, splitLengths);
    const auto variadicSplit = std::make_shared<ov::opset1::VariadicSplit>(deqBefore, constantAxis, constantLengths);

    ov::ResultVector results;
    for (size_t i = 0; i < splitLengths.size(); ++i) {
        results.push_back(std::make_shared<ov::opset1::Result>(
            dequantizationAfter.empty() ? variadicSplit->output(i) : makeDequantization(variadicSplit->output(i), dequantizationAfter[i])));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "VariadicSplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
