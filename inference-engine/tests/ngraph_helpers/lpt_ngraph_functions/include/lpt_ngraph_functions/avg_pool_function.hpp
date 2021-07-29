// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class AvgPoolFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& inputShape,
        const bool addFQ,
        const std::vector<std::string>& additionalLayers,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& inputShape,
        const bool addFQ,
        const std::vector<std::string>& additionalLayers,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationEnd);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
