// Copyright (C) 2021 Intel Corporation
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
        const ngraph::Shape& inputShape,
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
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::vector<int64_t>& constantValues,
        const bool keepDims) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        const auto fakeQuantize = makeFakeQuantize(input, precision, fqOnData);

        const auto constant = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i32,
            ngraph::Shape{ constantValues.size() },
            constantValues);

        const auto reduce = std::make_shared<ReduceType>(fakeQuantize, constant, keepDims);
        reduce->set_friendly_name("Output");

        const auto result = std::make_shared<ngraph::opset1::Result>(reduce);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            "ReduceTransformation");

        return function;
    }

    template <typename ReduceType>
    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
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
