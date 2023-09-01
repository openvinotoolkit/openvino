// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "lpt_ov_models/common/dequantization_operations.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/interpolate.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class InterpolateFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::Shape& outputShape,
        const ov::op::v0::Interpolate::Attributes& interpAttrs,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::Shape& outputShape,
        const ov::op::v0::Interpolate::Attributes& interpAttrs);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::Shape& outputShape,
        const ov::op::v0::Interpolate::Attributes& interpAttrs,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter);

    // v4::Interpolate
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::Shape& outputShape,
        const ov::Shape& scalesShape,
        const ov::op::v4::Interpolate::InterpolateAttrs& interp4Attrs,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::Shape& outputShape,
        const ov::Shape& scalesShape,
        const ov::op::v4::Interpolate::InterpolateAttrs& interp4Attrs);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::Shape& outputShape,
        const ov::Shape& scalesShape,
        const ov::op::v4::Interpolate::InterpolateAttrs& interp4Attrs,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
