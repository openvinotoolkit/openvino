// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class BroadcastFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const bool v1,
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const Shape& tagetShape,
        const Shape& axesMapping,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
