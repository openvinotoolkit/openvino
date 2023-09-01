// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>


#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"



namespace ov {
namespace builder {
namespace subgraph {

class VariadicSplitFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type originalFunctionPrecision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::element::Type inputPrecision,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const std::vector<ov::builder::subgraph::DequantizationOperations>& dequantizationAfter,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
