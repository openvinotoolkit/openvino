// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "common/fake_quantize_on_data.hpp"
#include "low_precision/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class AvgPoolFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const bool addFQ,
        const std::string additionalLayer,
        const ngraph::element::Type lowPrecision,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const bool addFQ,
        const std::string additionalLayer,
        const ngraph::element::Type activationPrecision,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
