// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/split_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ngraph::Function> SplitFunction::getOriginal(
    const element::Type& precision,
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const int64_t splitedAxis,
    const size_t numSplits) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    auto dequantizationStructure = dequantization;
    dequantizationStructure.multiply.outPrecision = precision;
    const auto dequantizationOp = makeDequantization(input, dequantization);
    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const auto split = std::make_shared<ngraph::opset1::Split>(dequantizationOp, constant, numSplits);

    ngraph::ResultVector results;
    for (size_t i = 0; i < numSplits; ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(i)));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitFunction");
}

std::shared_ptr<ngraph::Function> SplitFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::PartialShape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    int64_t splitedAxis, size_t numSplit) {
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

    auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const std::shared_ptr<ngraph::opset1::Split> split = std::make_shared<ngraph::opset1::Split>(fq, constant, numSplit);

    ngraph::ResultVector results;
    for (size_t i = 0; i < numSplit; ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(i)));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitFunction");
}

std::shared_ptr<ngraph::Function> SplitFunction::getReference(
    const element::Type& precision,
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type inputPrecision,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const std::vector<ngraph::builder::subgraph::DequantizationOperations>& dequantizationAfter,
    const int64_t splitedAxis,
    const size_t numSplit) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const auto split = std::make_shared<ngraph::opset1::Split>(deqBefore, constant, numSplit);

    ngraph::ResultVector results;
    for (size_t i = 0; i < numSplit; ++i) {
        if (!dequantizationAfter.empty()) {
            auto dequantizationStructure = dequantizationAfter[i];
            if (!dequantizationStructure.multiply.empty()) {
                dequantizationStructure.multiply.outPrecision = precision;
            }
            results.push_back(std::make_shared<ngraph::opset1::Result>(makeDequantization(split->output(i), dequantizationAfter[i])));
        } else {
            results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(i)));
        }
    }

    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
