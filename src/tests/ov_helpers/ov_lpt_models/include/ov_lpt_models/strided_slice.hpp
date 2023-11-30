// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <low_precision/layer_transformation.hpp>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class StridedSliceFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const std::vector<int64_t>& begin,
        const std::vector<int64_t>& end,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& beginMask,
        const std::vector<int64_t>& endMask,
        const std::vector<int64_t>& newAxisMask,
        const std::vector<int64_t>& shrinkAxisMask,
        const std::vector<int64_t>& elipsisMask);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
        const std::vector<int64_t>& begin,
        const std::vector<int64_t>& end,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& beginMask,
        const std::vector<int64_t>& endMask,
        const std::vector<int64_t>& newAxisMask,
        const std::vector<int64_t>& shrinkAxisMask,
        const std::vector<int64_t>& elipsisMask);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const std::vector<int64_t>& begin,
        const std::vector<int64_t>& end,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& beginMask,
        const std::vector<int64_t>& endMask,
        const std::vector<int64_t>& newAxisMask,
        const std::vector<int64_t>& shrinkAxisMask,
        const std::vector<int64_t>& elipsisMask,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
