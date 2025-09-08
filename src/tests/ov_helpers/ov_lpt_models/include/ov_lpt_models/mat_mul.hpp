// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/constant.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class MatMulFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape inputShape,
        const float low,
        const float high);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape1,
        const ov::PartialShape& inputShape2,
        const bool transpose1,
        const bool transpose2,
        const bool signedWeights,
        const bool bias,
        const bool perChannelWeightsDequantization,
        const bool relu,
        const bool fq);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape1,
        const FakeQuantizeOnData& fqOnData1,
        const ov::Shape& inputShape2,
        const FakeQuantizeOnData& fqOnData2,
        const bool requantization = false);

    static std::shared_ptr<ov::Model> getOriginal(const ov::element::Type netPrecision,
                                                  const ov::PartialShape& inputShape1,
                                                  const ov::element::Type precisionBeforeDequantization1,
                                                  const DequantizationOperations& dequantization1,
                                                  const ov::PartialShape& inputShape2,
                                                  const ov::element::Type precisionBeforeDequantization2,
                                                  const DequantizationOperations& dequantization2);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const DequantizationOperations& deqOnData,
        const Constant& weights,
        const FakeQuantizeOnWeights& fqOnWeights,
        const DequantizationOperations& deqOnWeights);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape1,
        const ov::element::Type precisionBeforeDequantization1,
        const DequantizationOperations& dequantization1,
        const ov::PartialShape& inputShape2,
        const ov::element::Type precisionBeforeDequantization2,
        const DequantizationOperations& dequantization2,
        const DequantizationOperations& resultDequantization);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const Constant& weights,
        const DequantizationOperations& resultDequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData,
        const Constant& weights,
        const FakeQuantizeOnDataWithConstant& fqOnWeights,
        const DequantizationOperations& deqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
