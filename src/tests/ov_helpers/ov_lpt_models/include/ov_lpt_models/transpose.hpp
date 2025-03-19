// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class TransposeFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const std::vector<int>& transposeConstValues,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const std::vector<int>& transposeConstValues,
        const ov::element::Type precisionBeforeFq,
        const FakeQuantizeOnData& fqOnData);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const std::vector<int>& transposeConstValues,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
