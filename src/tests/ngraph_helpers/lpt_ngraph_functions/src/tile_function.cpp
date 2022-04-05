// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/tile_function.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> TileFunction::getOriginal(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    std::vector<size_t> repeatsValue(inputShape.size());
    std::iota(repeatsValue.begin(), repeatsValue.end(), 1);
    const auto repeats = std::make_shared<opset1::Constant>(element::i32, Shape{repeatsValue.size()}, repeatsValue);
    const std::shared_ptr<Node> tile = std::make_shared<ngraph::opset1::Tile>(dequantizationOp, repeats);
    tile->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(tile) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "TileFunction");
}

std::shared_ptr<ngraph::Function> TileFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);

    const std::shared_ptr<Node> fq = fakeQuantize.empty() ? input :
                                     ngraph::builder::makeFakeQuantize(
                                             input,
                                             precision,
                                             fakeQuantize.quantizationLevel,
                                             fakeQuantize.constantShape,
                                             fakeQuantize.inputLowValues,
                                             fakeQuantize.inputHighValues,
                                             fakeQuantize.outputLowValues,
                                             fakeQuantize.outputHighValues);

    std::vector<size_t> repeatsValue(inputShape.size());
    std::iota(repeatsValue.begin(), repeatsValue.end(), 1);
    const auto repeats = std::make_shared<opset1::Constant>(element::i32, Shape{repeatsValue.size()}, repeatsValue);
    const std::shared_ptr<Node> tile = std::make_shared<ngraph::opset1::Tile>(fq, repeats);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(tile) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "TileFunction");
}

std::shared_ptr<ngraph::Function> TileFunction::getReference(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);

    std::vector<size_t> repeatsValue(inputShape.size());
    std::iota(repeatsValue.begin(), repeatsValue.end(), 1);
    const auto repeats = std::make_shared<opset1::Constant>(element::i32, Shape{repeatsValue.size()}, repeatsValue);
    const std::shared_ptr<Node> tile = std::make_shared<ngraph::opset1::Tile>(dequantizationOpBefore, repeats);
    const std::shared_ptr<Node> deqquantizationOpAfter = makeDequantization(tile, dequantizationAfter);
    deqquantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(deqquantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "TileFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
