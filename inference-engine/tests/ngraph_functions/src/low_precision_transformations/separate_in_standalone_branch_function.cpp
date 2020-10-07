// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/separate_in_standalone_branch_function.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> SeparateInStandaloneBranchFunction::getOriginal(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const size_t numberOfOperations) {
        auto input = std::make_shared<ngraph::op::v0::Parameter>(inputPrecision, inputShape);
        input->set_friendly_name("input");
        auto deq = ngraph::builder::subgraph::makeDequantization(input, dequantization);

        ngraph::ResultVector results;
        std::shared_ptr<ngraph::Node> targetOp;
        for (size_t i = 0; i < numberOfOperations; ++i) {
            targetOp = std::make_shared<ngraph::opset1::Clamp>(deq, 0.f, 6.f);
            targetOp->set_friendly_name("Clamp" + std::to_string(i));

            auto result = std::make_shared<ngraph::opset1::Result>(targetOp);
            result->set_friendly_name("result" + std::to_string(i));
            results.push_back(result);
        }

        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SeparateInStandaloneBranchFunction");
    }

    std::shared_ptr<ngraph::Function> SeparateInStandaloneBranchFunction::getReference(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const size_t numberOfOperations,
        const size_t indexOfTargetOperation) {
        auto input = std::make_shared<ngraph::op::v0::Parameter>(inputPrecision, inputShape);
        input->set_friendly_name("input");
        auto deq = ngraph::builder::subgraph::makeDequantization(input, dequantization);

        auto targetOp = std::make_shared<ngraph::opset1::Clamp>(deq, 0.f, 6.f);
        targetOp->set_friendly_name("Clamp" + std::to_string(indexOfTargetOperation));

        const auto dequantizeNodes = ngraph::pass::low_precision::NetworkHelper::getDequantization(targetOp);
        std::string postfix = numberOfOperations > 1 ? "/" + std::to_string(indexOfTargetOperation) : "";

        dequantizeNodes.convert->set_friendly_name(dequantizeNodes.convert->get_friendly_name() + postfix);
        dequantizeNodes.subtract->set_friendly_name(dequantizeNodes.subtract->get_friendly_name() + postfix);
        dequantizeNodes.subtractConstant->set_friendly_name(dequantizeNodes.subtractConstant->get_friendly_name() + postfix);
        dequantizeNodes.multiply->set_friendly_name(dequantizeNodes.multiply->get_friendly_name() + postfix);
        dequantizeNodes.multiplyConstant->set_friendly_name(dequantizeNodes.multiplyConstant->get_friendly_name() + postfix);

        const auto result = std::make_shared<ngraph::opset1::Result>(targetOp);
        result->set_friendly_name("result");
        return std::make_shared<ngraph::Function>(
            result,
            ngraph::ParameterVector{ input },
            "SeparateInStandaloneBranchFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
