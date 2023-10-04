// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/gather.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> GatherFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const std::vector<size_t>& gatherIndicesShape,
    const std::vector<int>& gatherIndicesValues,
    const std::vector<int>& axis,
    const int64_t batch_dims,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const int opset_version) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const auto indicesNode = std::make_shared<ngraph::opset1::Constant>(
        ngraph::element::i64,
        ngraph::Shape(gatherIndicesShape),
        gatherIndicesValues);
    const auto axisNode = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ axis.size() }, axis);
    std::shared_ptr<Node> gather;
    if (opset_version == 7) {
        gather = std::make_shared<ngraph::opset7::Gather>(dequantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 8) {
        gather = std::make_shared<ngraph::opset8::Gather>(dequantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 1) {
        gather = std::make_shared<ngraph::opset1::Gather>(dequantizationOp, indicesNode, axisNode);
    } else {
        throw std::runtime_error("Unknown opset version");
    }
    gather->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(gather) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "GatherFunction");
}

std::shared_ptr<ngraph::Function> GatherFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const std::vector<size_t>& gatherIndicesShape,
    const std::vector<int>& gatherIndicesValues,
    const std::vector<int>& axis,
    const int64_t batch_dims,
    const ngraph::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData,
    const int opset_version) {

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);

    const auto indicesNode = std::make_shared<ngraph::opset1::Constant>(
        ngraph::element::i64,
        ngraph::Shape(gatherIndicesShape),
        gatherIndicesValues);
    const auto axisNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ axis.size() }, axis);

    std::shared_ptr<Node> gather;
    if (opset_version == 7) {
        gather = std::make_shared<ngraph::opset7::Gather>(quantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 8) {
        gather = std::make_shared<ngraph::opset8::Gather>(quantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 1) {
        gather = std::make_shared<ngraph::opset1::Gather>(quantizationOp, indicesNode, axisNode);
    } else {
        throw std::runtime_error("Unknown opset version");
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(gather) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "GatherFunction");
}

std::shared_ptr<ngraph::Function> GatherFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const std::vector<size_t>& gatherIndicesShape,
    const std::vector<int>& gatherIndicesValues,
    const std::vector<int>& axis,
    const int64_t batch_dims,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int opset_version) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    const auto indicesNode = std::make_shared<ngraph::opset1::Constant>(
        ngraph::element::i64,
        ngraph::Shape(gatherIndicesShape),
        gatherIndicesValues);
    const auto axisNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ axis.size() }, axis);

    std::shared_ptr<Node> gather;
    if (opset_version == 7) {
        gather = std::make_shared<ngraph::opset7::Gather>(quantizationOpBefore, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 8) {
        gather = std::make_shared<ngraph::opset8::Gather>(quantizationOpBefore, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 1) {
        gather = std::make_shared<ngraph::opset1::Gather>(quantizationOpBefore, indicesNode, axisNode);
    } else {
        throw std::runtime_error("Unknown opset version");
    }

    if (quantizationOpBefore->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*quantizationOpBefore) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }
    if (gather->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*gather) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }

    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(gather, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "GatherFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
