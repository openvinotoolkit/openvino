// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class StandardizeDequantizationFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations dequantization);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::Shape& inputShape,
        const std::vector<float>& axes,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const ngraph::element::Type precisionAfterOperation);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
