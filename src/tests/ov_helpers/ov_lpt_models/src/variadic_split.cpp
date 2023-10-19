// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_lpt_models/variadic_split.hpp"

#include "ov_models/builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> VariadicSplitFunction::getOriginal(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

        const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
        const auto constantAxis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
        const auto constantLengths = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ splitLengths.size() }, splitLengths);
        const std::shared_ptr<Node> variadicSplit = std::make_shared<ngraph::opset1::VariadicSplit>(dequantizationOp, constantAxis, constantLengths);

        ngraph::ResultVector results;
        for (size_t i = 0; i < splitLengths.size(); ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(variadicSplit->output(i)));
        }
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "VariadicSplitFunction");
    }

std::shared_ptr<ngraph::Function> VariadicSplitFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::PartialShape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    const int64_t splitedAxis,
    const std::vector<size_t>& splitLengths) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fq = fakeQuantize.empty() ? nullptr :
        ngraph::builder::makeFakeQuantize(
            input,
            originalFunctionPrecision,
            fakeQuantize.quantizationLevel,
            fakeQuantize.constantShape,
            fakeQuantize.inputLowValues,
            fakeQuantize.inputHighValues,
            fakeQuantize.outputLowValues,
            fakeQuantize.outputHighValues);

    const auto constantAxis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const auto constantLengths = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ splitLengths.size() }, splitLengths);
    const std::shared_ptr<Node> variadicSplit =
        std::make_shared<ngraph::opset1::VariadicSplit>(fakeQuantize.empty() ? input : fq, constantAxis, constantLengths);

    ngraph::ResultVector results;
    for (size_t i = 0; i < splitLengths.size(); ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(variadicSplit->output(i)));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "VariadicSplitFunction");
}

std::shared_ptr<ngraph::Function> VariadicSplitFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type inputPrecision,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const std::vector<ngraph::builder::subgraph::DequantizationOperations>& dequantizationAfter,
    const int64_t splitedAxis,
    const std::vector<size_t>& splitLengths) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);

    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto constantAxis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const auto constantLengths = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ splitLengths.size() }, splitLengths);
    const auto variadicSplit = std::make_shared<ngraph::opset1::VariadicSplit>(deqBefore, constantAxis, constantLengths);

    ngraph::ResultVector results;
    for (size_t i = 0; i < splitLengths.size(); ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(
            dequantizationAfter.empty() ? variadicSplit->output(i) : makeDequantization(variadicSplit->output(i), dequantizationAfter[i])));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "VariadicSplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
