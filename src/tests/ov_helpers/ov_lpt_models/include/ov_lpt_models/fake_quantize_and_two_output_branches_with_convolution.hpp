// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <low_precision/layer_transformation.hpp>
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData,
        const FakeQuantizeOnWeights fqOnWeights1,
        FakeQuantizeOnWeights fqOnWeights2);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::pass::low_precision::LayerTransformation::Params& params,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const ov::element::Type precisionBeforeOp,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOp,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter1,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
