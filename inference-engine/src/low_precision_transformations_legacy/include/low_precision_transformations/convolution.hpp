// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include "low_precision_transformations/weightable_layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(ConvolutionTransformation) : public WeightableLayerTransformation {
public:
    ConvolutionTransformation(const Params& params) : WeightableLayerTransformation(params) {}
    ~ConvolutionTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;

private:
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
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
