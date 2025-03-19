// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ov_lpt_models/common/add.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/multiply.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class SubtractMultiplyToMultiplyAddFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const ov::element::Type precisionAfterDequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::element::Type precision,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnData);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const ov::element::Type precisionAfterDequantization,
        const ov::builder::subgraph::Multiply& multiply,
        const ov::builder::subgraph::Add& add);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
