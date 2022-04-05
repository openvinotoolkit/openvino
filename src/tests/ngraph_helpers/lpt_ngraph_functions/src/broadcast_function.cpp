// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/broadcast_function.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

static std::shared_ptr<Node> makeBroadcast(std::shared_ptr<Node> broadcastInput, size_t opset, std::string mode, std::shared_ptr<Node> modelInput) {
    std::shared_ptr<Node> outputShapeNode;
    const auto inputPartialShape = modelInput->get_output_partial_shape(0);
    if (inputPartialShape.is_static()) {
        auto inputShape = inputPartialShape.get_shape();
        auto outputShape = inputShape;
        outputShape.back() = 10;
        outputShapeNode =
                std::make_shared<opset1::Constant>(element::i64, Shape{outputShape.size()}, std::vector<size_t>(outputShape)); // {..., 1} -> {..., 10}
    } else {
        const auto sliceStart = std::make_shared<opset1::Constant>(element::i64, Shape{1}, std::vector<size_t>{0});
        const auto sliceStop = std::make_shared<opset1::Constant>(element::i64, Shape{1}, std::vector<size_t>{inputPartialShape.size() - 1});
        const auto sliceStep = std::make_shared<opset1::Constant>(element::i64, Shape{1}, std::vector<size_t>{1});

        outputShapeNode = std::make_shared<opset1::Concat>(OutputVector{
            std::make_shared<opset8::Slice>(std::make_shared<opset1::ShapeOf>(modelInput), sliceStart, sliceStop, sliceStep),
            std::make_shared<opset1::Constant>(element::i64, Shape{1}, std::vector<size_t>{10})}, 0); // {..., 1} -> {..., 10}
        //std::make_shared<opset1::Constant>(element::i32, Shape{outputShape.size()}, std::vector<size_t>(outputShape));
    }
    std::shared_ptr<Node> broadcast;
    if (mode == "numpy" || mode == "bidirectional")  {
        switch (opset) {
            case 1:
                broadcast = std::make_shared<ngraph::opset1::Broadcast>(broadcastInput, outputShapeNode, ov::op::AutoBroadcastType::NUMPY);
                break;
            case 3:
                broadcast = std::make_shared<ngraph::opset3::Broadcast>(
                    broadcastInput,
                    outputShapeNode,
                    mode == "numpy" ? ov::op::BroadcastType::NUMPY : ov::op::BroadcastType::BIDIRECTIONAL);
                break;
            default:
                THROW_TRANSFORMATION_EXCEPTION;
        }
    } else {
        std::vector<int> mapping(inputPartialShape.rank().get_length());
        std::iota(mapping.begin(), mapping.end(), 0);
        switch (opset) {
            case 1:
                broadcast = std::make_shared<ngraph::opset1::Broadcast>(
                        broadcastInput,
                        outputShapeNode,
                        std::make_shared<opset1::Constant>(element::i32, Shape{mapping.size()}, mapping),
                        ov::op::AutoBroadcastType::EXPLICIT);
                break;
            case 3:
                broadcast = std::make_shared<ngraph::opset3::Broadcast>(
                        broadcastInput,
                        outputShapeNode,
                        std::make_shared<opset1::Constant>(element::i32, Shape{mapping.size()}, mapping),
                        ov::op::BroadcastType::EXPLICIT);
                break;
            default:
                THROW_TRANSFORMATION_EXCEPTION;
        }
    }
    return broadcast;
}

std::shared_ptr<ngraph::Function> BroadcastFunction::getOriginal(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const size_t opset,
        const std::string mode) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    std::shared_ptr<Node> broadcast;
    broadcast = makeBroadcast(dequantizationOp, opset, mode, input);
    broadcast->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(broadcast) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "BroadcastFunction");
}

std::shared_ptr<ngraph::Function> BroadcastFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const size_t opset,
        const std::string mode) {
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

    std::shared_ptr<Node> broadcast;
    broadcast = makeBroadcast(fq, opset, mode, input);
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(broadcast) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "BroadcastFunction");
}

std::shared_ptr<ngraph::Function> BroadcastFunction::getReference(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const size_t opset,
        const std::string mode) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);

    std::shared_ptr<Node> broadcast;
    broadcast = makeBroadcast(dequantizationOpBefore, opset, mode, input);
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(broadcast, dequantizationAfter);
    dequantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "BroadcastFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
