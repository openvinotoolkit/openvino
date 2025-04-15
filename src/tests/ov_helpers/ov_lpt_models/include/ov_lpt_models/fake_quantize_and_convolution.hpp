// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/multiply.hpp"
#include "ov_lpt_models/common/reshape.hpp"
#include "ov_lpt_models/common/transpose.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FakeQuantizeAndConvolutionFunction {
public:
    // TODO: move to ConvolutionFunction
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights);

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
        const DequantizationOperations::Convert& convertOnData,
        const DequantizationOperations& dequantizationOnData,
        const Constant& constantOnWeights,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const DequantizationOperations::Convert& convertOnWeights,
        const DequantizationOperations& dequantizationOnWeights,
        const DequantizationOperations& dequantizationAfter,
        const std::string operation = "Convolution");

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
        const DequantizationOperations::Convert& convertOnData,
        const DequantizationOperations& dequantizationOnData,
        const Constant& constantOnWeights,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const DequantizationOperations::Convert& convertOnWeights,
        const DequantizationOperations& dequantizationOnWeights,
        const Reshape& reshape1,
        const DequantizationOperations::Multiply& multiply,
        const Transpose& transpose,
        const Reshape& reshape2,
        const DequantizationOperations& dequantizationAfter,
        const std::string operation = "Convolution",
        bool multiplyAfter = false);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
