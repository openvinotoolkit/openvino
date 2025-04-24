// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>


#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class NormalizeL2Function {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const std::pair<ov::PartialShape, ov::Shape>& shapes,
        const ov::element::Type precisionOnActivation,
        const std::vector<uint64_t>& axes,
        const bool fuseMultiply,
        const bool shift);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::element::Type inputPrecision,
        const ov::PartialShape& shape,
        const ov::op::EpsMode& epsMode,
        const std::vector<size_t>& axes,
        const ov::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::element::Type inputPrecision,
        const ov::PartialShape& shape,
        const ov::op::EpsMode& epsMode,
        const std::vector<size_t>& axes,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
