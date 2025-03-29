// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"



namespace ov {
namespace builder {
namespace subgraph {

class SplitFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type& precision,
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const size_t numSplits);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type originalFunctionPrecision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const int64_t splitedAxis,
        const size_t numSplit);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type& precision,
        const ov::PartialShape& inputShape,
        const ov::element::Type inputPrecision,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const std::vector<ov::builder::subgraph::DequantizationOperations>& dequantizationAfter,
        const int64_t splitedAxis,
        const size_t numSplits);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
