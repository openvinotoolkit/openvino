// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/gather.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> GatherFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<size_t>& gatherIndicesShape,
    const std::vector<int>& gatherIndicesValues,
    const std::vector<int>& axis,
    const int64_t batch_dims,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const int opset_version) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const auto indicesNode = std::make_shared<ov::opset1::Constant>(
        ov::element::i64,
        ov::Shape(gatherIndicesShape),
        gatherIndicesValues);
    const auto axisNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ axis.size() }, axis);
    std::shared_ptr<Node> gather;
    if (opset_version == 7) {
        gather = std::make_shared<ov::opset7::Gather>(dequantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 8) {
        gather = std::make_shared<ov::opset8::Gather>(dequantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 1) {
        gather = std::make_shared<ov::opset1::Gather>(dequantizationOp, indicesNode, axisNode);
    } else {
        throw std::runtime_error("Unknown opset version");
    }
    gather->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(gather) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "GatherFunction");
}

std::shared_ptr<ov::Model> GatherFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<size_t>& gatherIndicesShape,
    const std::vector<int>& gatherIndicesValues,
    const std::vector<int>& axis,
    const int64_t batch_dims,
    const ov::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData,
    const int opset_version) {

    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);

    const auto indicesNode = std::make_shared<ov::opset1::Constant>(
        ov::element::i64,
        ov::Shape(gatherIndicesShape),
        gatherIndicesValues);
    const auto axisNode = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ axis.size() }, axis);

    std::shared_ptr<Node> gather;
    if (opset_version == 7) {
        gather = std::make_shared<ov::opset7::Gather>(quantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 8) {
        gather = std::make_shared<ov::opset8::Gather>(quantizationOp, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 1) {
        gather = std::make_shared<ov::opset1::Gather>(quantizationOp, indicesNode, axisNode);
    } else {
        throw std::runtime_error("Unknown opset version");
    }

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(gather) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "GatherFunction");
}

std::shared_ptr<ov::Model> GatherFunction::getReference(
    const ov::PartialShape& inputShape,
    const std::vector<size_t>& gatherIndicesShape,
    const std::vector<int>& gatherIndicesValues,
    const std::vector<int>& axis,
    const int64_t batch_dims,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int opset_version) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    const auto indicesNode = std::make_shared<ov::opset1::Constant>(
        ov::element::i64,
        ov::Shape(gatherIndicesShape),
        gatherIndicesValues);
    const auto axisNode = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ axis.size() }, axis);

    std::shared_ptr<Node> gather;
    if (opset_version == 7) {
        gather = std::make_shared<ov::opset7::Gather>(quantizationOpBefore, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 8) {
        gather = std::make_shared<ov::opset8::Gather>(quantizationOpBefore, indicesNode, axisNode, batch_dims);
    } else if (opset_version == 1) {
        gather = std::make_shared<ov::opset1::Gather>(quantizationOpBefore, indicesNode, axisNode);
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

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "GatherFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
