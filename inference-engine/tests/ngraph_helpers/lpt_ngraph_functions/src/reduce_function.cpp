// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"

#include "ngraph/opsets/opset1.hpp"

#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/reduce_function.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

    std::shared_ptr<Node> getReduceNodeByType(
        const std::string type,
        const std::shared_ptr<Node> parent,
        const std::shared_ptr<opset1::Constant> constant,
        const element::Type precision,
        const bool keepDims) {
        if (type == "Mean") {
            return std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::ReduceMean>>(
                std::vector<element::Type>{ precision, constant->get_element_type() },
                std::vector<element::Type>{ precision },
                ngraph::op::TemporaryReplaceOutputType(parent, precision).get(),
                ngraph::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
                keepDims);
        } else if (type == "Sum") {
            return std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::ReduceSum>>(
                std::vector<element::Type>{ precision, constant->get_element_type() },
                std::vector<element::Type>{ precision },
                ngraph::op::TemporaryReplaceOutputType(parent, precision).get(),
                ngraph::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
                keepDims);
        } else if (type == "Max") {
            return std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::ReduceMax>>(
                std::vector<element::Type>{ precision, constant->get_element_type() },
                std::vector<element::Type>{ precision },
                ngraph::op::TemporaryReplaceOutputType(parent, precision).get(),
                ngraph::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
                keepDims);
        } else if (type == "Min") {
            return std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::ReduceMin>>(
                std::vector<element::Type>{ precision, constant->get_element_type() },
                std::vector<element::Type>{ precision },
                ngraph::op::TemporaryReplaceOutputType(parent, precision).get(),
                ngraph::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
                keepDims);
        } else {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected Reduce type";
        }
    }

    std::string getFunctionNameByReduceType(std::string reduceType) {
        if (reduceType == "Mean") {
            return "ReduceMeanFunction";
        } else if (reduceType == "Sum") {
            return "ReduceSumFunction";
        } else if (reduceType == "Max") {
            return "ReduceMaxFunction";
        } else if (reduceType == "Min") {
            return "ReduceMinFunction";
        } else {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected Reduce type";
        }
    }

    std::shared_ptr<ngraph::Function> ReduceFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const std::string reduceType,
        const bool keepDims) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        const auto dequantization = makeDequantization(input, dequantizationBefore);

        const auto constant = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i32,
            ngraph::Shape{ constantValues.size() },
            constantValues);

        const auto reducePrecision = dequantization->get_output_element_type(0);
        const std::shared_ptr<Node> reduce = getReduceNodeByType(reduceType, dequantization, constant, reducePrecision, keepDims);

        reduce->set_friendly_name("Output");
        const auto result = std::make_shared<ngraph::opset1::Result>(reduce);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            getFunctionNameByReduceType(reduceType));

        return function;
    }

    std::shared_ptr<ngraph::Function> ReduceFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::vector<int64_t>& constantValues,
        const std::string reduceType,
        const bool keepDims) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        const auto dequantization = makeFakeQuantize(input, precision, fqOnData);

        const auto constant = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i32,
            ngraph::Shape{ constantValues.size() },
            constantValues);

        const auto reducePrecision = dequantization->get_output_element_type(0);
        const std::shared_ptr<Node> reduce = getReduceNodeByType(reduceType, dequantization, constant, reducePrecision, keepDims);

        reduce->set_friendly_name("Output");
        const auto result = std::make_shared<ngraph::opset1::Result>(reduce);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            getFunctionNameByReduceType(reduceType));

        return function;
    }

    std::shared_ptr<ngraph::Function> ReduceFunction::getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const std::string reduceType,
        const bool keepDims,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        const auto dequantization = makeDequantization(input, dequantizationBefore);

        const auto constant = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i32,
            ngraph::Shape{ constantValues.size() },
            constantValues);

        const std::shared_ptr<Node> reduce = getReduceNodeByType(reduceType, dequantization, constant, precisionAfterOperation, keepDims);
        std::shared_ptr<Node> lastOperation = makeDequantization(reduce, dequantizationAfter);

        lastOperation->set_friendly_name("Output");
        const auto result = std::make_shared<ngraph::opset1::Result>(lastOperation);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            getFunctionNameByReduceType(reduceType));

        return function;
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
