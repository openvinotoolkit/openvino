// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/constant.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape inputShape,
        const float low,
        const float high);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape inputShape1,
        const ngraph::PartialShape inputShape2,
        const bool transpose1,
        const bool transpose2);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape1,
        const FakeQuantizeOnData& fqOnData1,
        const ngraph::Shape& inputShape2,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const element::Type netPrecision,
        const ngraph::PartialShape& inputShape1,
        const ngraph::element::Type precisionBeforeDequantization1,
        const DequantizationOperations& dequantization1,
        const ngraph::PartialShape& inputShape2,
        const ngraph::element::Type precisionBeforeDequantization2,
        const DequantizationOperations& dequantization2);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& deqOnData,
        const Constant& weights,
        const FakeQuantizeOnWeights& fqOnWeights,
        const DequantizationOperations& deqOnWeights);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape1,
        const ngraph::element::Type precisionBeforeDequantization1,
        const DequantizationOperations& dequantization1,
        const ngraph::PartialShape& inputShape2,
        const ngraph::element::Type precisionBeforeDequantization2,
        const DequantizationOperations& dequantization2,
        const DequantizationOperations& resultDequantization);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const Constant& weights,
        const DequantizationOperations& resultDequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData,
        const Constant& weights,
        const FakeQuantizeOnDataWithConstant& fqOnWeights,
        const DequantizationOperations& deqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
