// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/common/convolution.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

Convolution::Convolution() {
}

Convolution::Convolution(
    const DequantizationOperations::Subtract zeroPointOnActivations,
    const Constant& constantOnWeights,
    const DequantizationOperations& dequantizationOnWeights) :
    zeroPointOnActivations(zeroPointOnActivations),
    constantOnWeights(constantOnWeights),
    dequantizationOnWeights(dequantizationOnWeights) {
}

bool Convolution::empty() const {
    return zeroPointOnActivations.empty() && constantOnWeights.empty() && dequantizationOnWeights.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
