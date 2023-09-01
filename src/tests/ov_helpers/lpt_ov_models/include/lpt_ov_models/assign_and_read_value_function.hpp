// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>

#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class AssignAndReadValueFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const element::Type& inputPrecision,
        const ov::element::Type precisionBeforeDequantization,
        const size_t opsetVersion,
        const bool FQAfterReadValue,
        const std::vector<float>& constantValue,
        const ov::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type originalFunctionPrecision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const size_t opsetVersion);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const element::Type& inputPrecision,
        const ov::element::Type precisionBeforeDequantization,
        const size_t opsetVersion,
        const bool FQAfterReadValue,
        const std::vector<float>& constantValue,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
