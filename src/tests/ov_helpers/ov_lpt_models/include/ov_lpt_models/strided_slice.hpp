// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <low_precision/layer_transformation.hpp>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class StridedSliceFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const std::vector<int64_t>& begin,
        const std::vector<int64_t>& end,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& beginMask,
        const std::vector<int64_t>& endMask,
        const std::vector<int64_t>& newAxisMask,
        const std::vector<int64_t>& shrinkAxisMask,
        const std::vector<int64_t>& elipsisMask,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter = {});

    // Variant where begin, end and strides are graph inputs (Parameters).
    // The PartialShape arguments describe the shape of each parameter.
    // dequantizationBefore is applied to the data input before StridedSlice;
    // dequantizationAfter is applied to the StridedSlice output (defaults to empty).
    static std::shared_ptr<ov::Model> getWithParamInputs(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::PartialShape& beginShape,
        const ov::PartialShape& endShape,
        const ov::PartialShape& stridesShape,
        const std::vector<int64_t>& beginMask,
        const std::vector<int64_t>& endMask,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter = {});

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
        const std::vector<int64_t>& begin,
        const std::vector<int64_t>& end,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& beginMask,
        const std::vector<int64_t>& endMask,
        const std::vector<int64_t>& newAxisMask,
        const std::vector<int64_t>& shrinkAxisMask,
        const std::vector<int64_t>& elipsisMask);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
