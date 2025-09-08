// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "common/dequantization_operations.hpp"
#include "common/add.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class MulAddToScaleshiftOrPowerFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        bool isDequantization,
        const ov::builder::subgraph::DequantizationOperations::Multiply& mulValues,
        const ov::builder::subgraph::Add& addValues);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        bool isDequantization,
        const ov::builder::subgraph::DequantizationOperations::Multiply& weightsValues,
        const ov::builder::subgraph::Add& biasesValues,
        const ov::element::Type precisionAfterOperation);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
