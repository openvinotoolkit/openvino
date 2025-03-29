// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/op/constant.hpp"

#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class MultiplyToGroupConvolutionFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::element::Type& precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const bool haveMultiplyWithNoConstBeforeDequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData,
        const Constant& constant,
        const bool parentHasOneConsumer = true);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::element::Type& precision,
        const std::shared_ptr<ov::op::v0::Constant>& weights,
        const std::shared_ptr<ov::op::v0::Constant>& biases,
        const ov::builder::subgraph::DequantizationOperations& dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
