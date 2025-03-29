// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_lpt_models/split.hpp"

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> SplitFunction::getOriginal(
    const ov::element::Type& precision,
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const int64_t splitedAxis,
    const size_t numSplits) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    auto dequantizationStructure = dequantization;
    dequantizationStructure.multiply.outPrecision = precision;
    const auto dequantizationOp = makeDequantization(input, dequantization);
    const auto constant = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{ }, splitedAxis);
    const auto split = std::make_shared<ov::opset1::Split>(dequantizationOp, constant, numSplits);

    ov::ResultVector results;
    for (size_t i = 0; i < numSplits; ++i) {
        results.push_back(std::make_shared<ov::opset1::Result>(split->output(i)));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SplitFunction");
}

std::shared_ptr<ov::Model> SplitFunction::getOriginal(
    const ov::element::Type originalFunctionPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    int64_t splitedAxis, size_t numSplit) {
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

    auto constant = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{ }, splitedAxis);
    const std::shared_ptr<ov::opset1::Split> split = std::make_shared<ov::opset1::Split>(fq, constant, numSplit);

    ov::ResultVector results;
    for (size_t i = 0; i < numSplit; ++i) {
        results.push_back(std::make_shared<ov::opset1::Result>(split->output(i)));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SplitFunction");
}

std::shared_ptr<ov::Model> SplitFunction::getReference(
    const ov::element::Type& precision,
    const ov::PartialShape& inputShape,
    const ov::element::Type inputPrecision,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const std::vector<ov::builder::subgraph::DequantizationOperations>& dequantizationAfter,
    const int64_t splitedAxis,
    const size_t numSplit) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto constant = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{ }, splitedAxis);
    const auto split = std::make_shared<ov::opset1::Split>(deqBefore, constant, numSplit);

    ov::ResultVector results;
    for (size_t i = 0; i < numSplit; ++i) {
        if (!dequantizationAfter.empty()) {
            auto dequantizationStructure = dequantizationAfter[i];
            if (!dequantizationStructure.multiply.empty()) {
                dequantizationStructure.multiply.outPrecision = precision;
            }
            results.push_back(std::make_shared<ov::opset1::Result>(makeDequantization(split->output(i), dequantizationAfter[i])));
        } else {
            results.push_back(std::make_shared<ov::opset1::Result>(split->output(i)));
        }
    }

    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
