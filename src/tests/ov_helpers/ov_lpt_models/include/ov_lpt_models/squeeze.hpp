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

class SqueezeFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::PartialShape& inputShape,
        const std::vector<float>& axes,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const std::vector<float>& axes);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::PartialShape& inputShape,
        const std::vector<float>& axes,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
