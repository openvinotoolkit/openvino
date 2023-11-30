// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MVNFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const element::Type precision,
        const ov::PartialShape& inputShape,
        const AxisSet& reductionAxes,
        const bool& normalizeVariance,
        const ov::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int opset_version);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const AxisSet& reductionAxes,
        const bool& normalizeVariance);

    static std::shared_ptr<ov::Model> getReference(
        const element::Type precision,
        const ov::PartialShape& inputShape,
        const AxisSet& reductionAxes,
        const bool& normalizeVariance,
        const ov::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const int opset_version);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
