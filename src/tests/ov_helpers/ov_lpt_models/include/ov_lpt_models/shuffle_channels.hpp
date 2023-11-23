// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <low_precision/layer_transformation.hpp>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ShuffleChannelsFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& deqBefore,
        const std::int64_t axis,
        const std::int64_t group);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::int64_t axis,
        const std::int64_t group);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& deqBefore,
        const std::int64_t axis,
        const std::int64_t group,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& deqAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
