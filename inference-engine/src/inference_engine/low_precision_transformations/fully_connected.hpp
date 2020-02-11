// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <ie_common.h>
#include <algorithm>
#include "low_precision_transformations/weightable_layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(FullyConnectedTransformation) : public WeightableLayerTransformation {
public:
    FullyConnectedTransformation(const Params& params) : WeightableLayerTransformation(params) {}
    ~FullyConnectedTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;

private:
    void calculateDequantizationForSymmetric(
        const CNNLayer& fullyConnected,
        const std::vector<float>& originalWeightsDequantizationScales,
        const std::vector<float>& originalWeightsDequantizationShifts,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts,
        std::vector<float>& biasesShifts) const;

    void calculateDequantizationForAsymmetric(
        const CNNLayer& fullyConnected,
        const std::vector<float>& dataZeroPoints,
        const std::vector<float>& originalWeightsDequantizationScales,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts) const;
};

}  // namespace details
}  // namespace InferenceEngine
