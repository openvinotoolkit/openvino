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

class GatherFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const std::vector<size_t>& gatherIndicesShape,
        const std::vector<int>& gatherIndicesValues,
        const std::vector<int>& axis,
        const int64_t batch_dims,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const int opset_version);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const std::vector<size_t>& gatherIndicesShape,
        const std::vector<int>& gatherIndicesValues,
        const std::vector<int>& axis,
        const int64_t batch_dims,
        const ov::element::Type precisionBeforeFq,
        const FakeQuantizeOnData& fqOnData,
        const int opset_version);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const std::vector<size_t>& gatherIndicesShape,
        const std::vector<int>& gatherIndicesValues,
        const std::vector<int>& axis,
        const int64_t batch_dims,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const int opset_version);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
