// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <string>

#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeAndConvolutionFunction {
public:
    // TODO: move to ConvolutionFunction
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
        const DequantizationOperations::Convert& convertOnData,
        const DequantizationOperations& dequantizationOnData,
        const Constant& constantOnWeights,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const DequantizationOperations::Convert& convertOnWeights,
        const DequantizationOperations& dequantizationOnWeights,
        const DequantizationOperations& dequantizationAfter,
        const std::string operation = "Convolution");
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
