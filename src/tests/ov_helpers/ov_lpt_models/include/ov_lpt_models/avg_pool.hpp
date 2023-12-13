// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class AvgPoolFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const bool addFQ,
        const std::vector<std::string>& additionalLayers,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type originalFunctionPrecision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const bool addFQ,
        const std::vector<std::string>& additionalLayers,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationEnd);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
