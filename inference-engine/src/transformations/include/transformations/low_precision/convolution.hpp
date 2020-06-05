// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include "weightable_layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API ConvolutionTransformation : public WeightableLayerTransformation {
public:
    ConvolutionTransformation(const Params& params);
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;

private:
#if 0 // TODO: LPT-TO-NGRAPH

    void calculateDequantizationForAsymmetric(
        const CNNLayer& convolution,
        const std::vector<float>& originalDataDequantizationScales,
        const std::vector<float>& originalDataDequantizationShifts,
        const std::vector<float>& dataZeroPoints,
        const std::vector<float>& originalWeightsDequantizationScales,
        const std::vector<float>& originalWeightsDequantizationShifts,
        const std::vector<float>& weightsZeroPoints,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts) const;

#endif
};

}// namespace low_precision
}// namespace pass
}// namespace ngraph