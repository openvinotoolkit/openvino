// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MVNFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const element::Type precision,
        const ngraph::PartialShape& inputShape,
        const AxisSet& reductionAxes,
        const bool& normalizeVariance,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int opset_version);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const AxisSet& reductionAxes,
        const bool& normalizeVariance);

    static std::shared_ptr<ngraph::Function> getReference(
        const element::Type precision,
        const ngraph::PartialShape& inputShape,
        const AxisSet& reductionAxes,
        const bool& normalizeVariance,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const int opset_version);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
