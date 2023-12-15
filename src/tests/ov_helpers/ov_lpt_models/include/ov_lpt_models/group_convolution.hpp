// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/op/constant.hpp"

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class GroupConvolutionFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::Shape& outputShape,
        const size_t groupCount,
        const int groupCalculationDimention,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ov::op::v0::Constant> weightsConst,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::Shape& outputShape,
        const size_t groupCount,
        const int groupCalculationDimention,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const bool addReshape = true,
        const bool addPrecisionPreserved = false);

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::PartialShape& outputShape,
        const size_t groupCount,
        const int calculatedDimention,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ov::op::v0::Constant> weightsConst,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ov::element::Type precisionAfterDequantization,
        const bool addReshape);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
