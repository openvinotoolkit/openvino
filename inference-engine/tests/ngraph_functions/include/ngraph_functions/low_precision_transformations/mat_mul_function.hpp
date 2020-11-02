// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape1,
        const FakeQuantizeOnData& fqOnData1,
        const ngraph::Shape& inputShape2,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape1,
        const ngraph::element::Type precisionBeforeDequantization1,
        const DequantizationOperations& dequantization1,
        const ngraph::Shape& inputShape2,
        const ngraph::element::Type precisionBeforeDequantization2,
        const DequantizationOperations& dequantization2);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ngraph::Shape& weightsConstShape,
        const std::vector<float>& weightsConstValues,
        const FakeQuantizeOnWeights& fqOnWeights);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape1,
        const ngraph::element::Type precisionBeforeDequantization1,
        const DequantizationOperations& dequantization1,
        const ngraph::Shape& inputShape2,
        const ngraph::element::Type precisionBeforeDequantization2,
        const DequantizationOperations& dequantization2,
        const DequantizationOperations& resultDequantization);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ngraph::element::Type weightsConstPrecision,
        const ngraph::Shape& weightsConstShape,
        const std::vector<float>& weightsConstValues,
        const DequantizationOperations& resultDequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData,
        const ngraph::Shape& weightsConstShape,
        const std::vector<float>& weightsConstValues,
        const FakeQuantizeOnWeights& fqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
