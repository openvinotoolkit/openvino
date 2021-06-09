// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <low_precision/layer_transformation.hpp>

#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ShuffleChannelsFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& deqBefore,
        const std::int64_t axis,
        const std::int64_t group);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::int64_t axis,
        const std::int64_t group);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& deqBefore,
        const std::int64_t axis,
        const std::int64_t group,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& deqAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
