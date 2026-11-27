// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class MoveDequantizationBeforeFunction {
public:
    // Two inputs of inputShape concatenated along axis, followed by dequantization.
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const std::int64_t axis,
        const ov::builder::subgraph::DequantizationOperations& dequantization);

    // A Concat along axis with dequantizationsBefore[i] in front of its i-th input.
    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const std::int64_t axis,
        const std::vector<ov::builder::subgraph::DequantizationOperations>& dequantizationsBefore);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
