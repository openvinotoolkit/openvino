// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class RoundWithToleranceFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantization);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
