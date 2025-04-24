// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <low_precision/layer_transformation.hpp>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/constant.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class ReduceFunction {
public:
    template <typename ReduceType>
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const bool keepDims) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        const auto dequantization = makeDequantization(input, dequantizationBefore);

        const auto constant = std::make_shared<ov::op::v0::Constant>(
            ov::element::i32,
            ov::Shape{ constantValues.size() },
            constantValues);

        const auto reducePrecision = dequantization->get_output_element_type(0);
        const std::shared_ptr<Node> reduce = std::make_shared<ov::op::TypeRelaxed<ReduceType>>(
            std::vector<ov::element::Type>{reducePrecision, constant->get_element_type()},
            std::vector<ov::element::Type>{reducePrecision},
            ov::op::TemporaryReplaceOutputType(dequantization, reducePrecision).get(),
            ov::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
            keepDims);

        reduce->set_friendly_name("Output");
        const auto result = std::make_shared<ov::op::v0::Result>(reduce);
        const auto function = std::make_shared<ov::Model>(
            ov::ResultVector{ result },
            ov::ParameterVector{ input },
            "ReduceTransformation");

        return function;
    }

    template <typename ReduceType>
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const ov::builder::subgraph::DequantizationOperations::Convert& convert,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const bool keepDims,
        ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        std::shared_ptr<ov::Node> parent = input;

        if (!fqOnData.empty()) {
            parent = makeFakeQuantize(parent, precision, fqOnData);
        }

        if (!convert.empty()) {
            parent = std::make_shared<ov::op::v0::Convert>(parent, convert.outPrecision);
        }

        if (!dequantizationBefore.empty()) {
            parent = makeDequantization(parent, dequantizationBefore);
        }

        const auto constant = std::make_shared<ov::op::v0::Constant>(
            ov::element::i32,
            ov::Shape{ constantValues.size() },
            constantValues);

        parent = std::make_shared<ReduceType>(parent, constant, keepDims);
        parent->set_friendly_name("Output");

        if (!dequantizationAfter.empty()) {
            parent = makeDequantization(parent, dequantizationAfter);
        }

        const auto result = std::make_shared<ov::op::v0::Result>(parent);
        const auto function = std::make_shared<ov::Model>(
            ov::ResultVector{ result },
            ov::ParameterVector{ input },
            "ReduceTransformation");

        return function;
    }

    template <typename ReduceType>
    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const std::vector<int64_t>& constantValues,
        const bool keepDims,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        const auto dequantization = makeDequantization(input, dequantizationBefore);

        const auto constant = std::make_shared<ov::op::v0::Constant>(
            ov::element::i32,
            ov::Shape{ constantValues.size() },
            constantValues);

        const std::shared_ptr<Node> reduce = std::make_shared<ov::op::TypeRelaxed<ReduceType>>(
            std::vector<ov::element::Type>{precisionAfterOperation, constant->get_element_type()},
            std::vector<ov::element::Type>{precisionAfterOperation},
            ov::op::TemporaryReplaceOutputType(dequantization, precisionAfterOperation).get(),
            ov::op::TemporaryReplaceOutputType(constant, constant->get_element_type()).get(),
            keepDims);
        std::shared_ptr<Node> lastOperation = makeDequantization(reduce, dequantizationAfter);

        lastOperation->set_friendly_name("Output");
        const auto result = std::make_shared<ov::op::v0::Result>(lastOperation);
        const auto function = std::make_shared<ov::Model>(
            ov::ResultVector{ result },
            ov::ParameterVector{ input },
            "ReduceTransformation");

        return function;
    }
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
