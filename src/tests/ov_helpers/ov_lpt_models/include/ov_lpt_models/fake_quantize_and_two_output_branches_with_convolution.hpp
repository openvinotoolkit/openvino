// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include <low_precision/layer_transformation.hpp>
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData,
        const FakeQuantizeOnWeights fqOnWeights1,
        FakeQuantizeOnWeights fqOnWeights2);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ov::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const ngraph::element::Type precisionBeforeOp,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOp,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
