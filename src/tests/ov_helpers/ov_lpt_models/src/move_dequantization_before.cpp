// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/move_dequantization_before.hpp"

#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MoveDequantizationBeforeFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const std::int64_t axis,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
    const auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{input1, input2}, axis);
    const auto deq = makeDequantization(concat, dequantization);

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(deq)},
                                       ov::ParameterVector{input1, input2},
                                       "MoveDequantizationBeforeFunction");
}

std::shared_ptr<ov::Model> MoveDequantizationBeforeFunction::getReference(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const std::int64_t axis,
    const std::vector<ov::builder::subgraph::DequantizationOperations>& dequantizationsBefore) {
    ov::ParameterVector inputs;
    ov::OutputVector concatInputs;
    for (const auto& dequantization : dequantizationsBefore) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        inputs.push_back(input);
        concatInputs.push_back(makeDequantization(input, dequantization));
    }
    const auto concat = std::make_shared<ov::op::v0::Concat>(concatInputs, axis);

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(concat)},
                                       inputs,
                                       "MoveDequantizationBeforeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
