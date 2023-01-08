// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <low_precision/layer_transformation.hpp>

#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ReduceFunction {
public:
    template <typename ReduceType>
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const bool keepDims) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        const auto dequantization = makeDequantization(input, dequantizationBefore);

        const auto constant = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i32,
            ngraph::Shape{ constantValues.size() },
            constantValues);

        const auto reducePrecision = dequantization->get_output_element_type(0);
        const std::shared_ptr<Node> reduce = std::make_shared<ngraph::op::TypeRelaxed<ReduceType>>(
            std::vector<element::Type>{ reducePrecision, constant->get_element_type() },
            std::vector<element::Type>{ reducePrecision },
            ngraph::op::TemporaryReplaceOutputType(dequantization, reducePrecision).get(),
            ngraph::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
            keepDims);

        reduce->set_friendly_name("Output");
        const auto result = std::make_shared<ngraph::opset1::Result>(reduce);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            "ReduceTransformation");

        return function;
    }

    template <typename ReduceType>
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const ngraph::builder::subgraph::DequantizationOperations::Convert& convert,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const bool keepDims,
        ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        std::shared_ptr<ngraph::Node> parent = input;

        if (!fqOnData.empty()) {
            parent = makeFakeQuantize(parent, precision, fqOnData);
        }

        if (!convert.empty()) {
            parent = std::make_shared<opset1::Convert>(parent, convert.outPrecision);
        }

        if (!dequantizationBefore.empty()) {
            parent = makeDequantization(parent, dequantizationBefore);
        }

        const auto constant = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i32,
            ngraph::Shape{ constantValues.size() },
            constantValues);

        parent = std::make_shared<ReduceType>(parent, constant, keepDims);
        parent->set_friendly_name("Output");

        if (!dequantizationAfter.empty()) {
            parent = makeDequantization(parent, dequantizationAfter);
        }

        const auto result = std::make_shared<ngraph::opset1::Result>(parent);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            "ReduceTransformation");

        return function;
    }

    template <typename ReduceType>
    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const bool keepDims,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        const auto dequantization = makeDequantization(input, dequantizationBefore);

        const auto constant = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i32,
            ngraph::Shape{ constantValues.size() },
            constantValues);

        const std::shared_ptr<Node> reduce = std::make_shared<ngraph::op::TypeRelaxed<ReduceType>>(
            std::vector<element::Type>{ precisionAfterOperation, constant->get_element_type() },
            std::vector<element::Type>{ precisionAfterOperation },
            ngraph::op::TemporaryReplaceOutputType(dequantization, precisionAfterOperation).get(),
            ngraph::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
            keepDims);
        std::shared_ptr<Node> lastOperation = makeDequantization(reduce, dequantizationAfter);

        lastOperation->set_friendly_name("Output");
        const auto result = std::make_shared<ngraph::opset1::Result>(lastOperation);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            "ReduceTransformation");

        return function;
    }
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
