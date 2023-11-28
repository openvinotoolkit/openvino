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

class MarkupAvgPoolPrecisionsFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::element::Type inputPrecision,
        const ov::Shape& inputShape,
        const bool addFQ,
        const std::string additionalLayer,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        // -1 - no Convolution
        const int convoutionBranch,
        // -1 - no FakeQuantize
        const int fakeQuantizeBranch);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type originalFunctionPrecision,
        const ov::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::element::Type inputPrecision,
        const ov::Shape& inputShape,
        const bool addFQ,
        const std::string additionalLayer,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
