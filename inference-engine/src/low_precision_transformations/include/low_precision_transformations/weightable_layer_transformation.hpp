// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "low_precision_transformations/transformation_context.hpp"
#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class PrecisionsInfo {
public:
    PrecisionsInfo(const Precision original, const Precision low) : original(original), low(low) {}
    const Precision original;
    const Precision low;
};

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(WeightableLayerTransformation) : public LayerTransformation{
public:
    WeightableLayerTransformation(const Params& params) : LayerTransformation(params) {}
    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
    bool isQuantized(const CNNLayer& layer) const noexcept override;

protected:
    void updateLayerBiases(
        TransformationContext& context,
        const CNNLayer& convolution,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts,
        std::vector<float>& biasesShifts) const;

    void updateWeights(
        const CNNLayerPtr fakeQuantize,
        std::vector<float>& outputLowValues,
        std::vector<float>& outputHighValues) const;

    void updateToSupportAsymmetricQuantization(
        TransformationContext& context,
        const CNNLayer& layer,
        const PrecisionsInfo& dataPrecisionsInfo,
        std::vector<float>& dataShifts,
        const PrecisionsInfo& weightsPrecisionsInfo,
        std::vector<float>& weightsShifts) const;

    void createAsymmetric(
        TransformationContext& context,
        const CNNLayer& parent,
        const CNNLayer& child,
        const PrecisionsInfo& precisionsInfo,
        const std::vector<float>& quantizationShifts,
        const bool onWeights) const;

    DataPrecision fillDequantizationsForWeightsPath(
        const CNNLayer& weightableLayer,
        const bool supportAsymmetricQuantization,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts) const;

    static bool isDepthwise(const CNNLayer& layer);

    void calculateDequantizationForSymmetric(
        const CNNLayer& weightableLayer,
        const std::vector<float>& originalDataDequantizationScales,
        const std::vector<float>& originalDataDequantizationShifts,
        const std::vector<float>& originalWeightsDequantizationScales,
        const std::vector<float>& originalWeightsDequantizationShifts,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts) const;
};

typedef std::shared_ptr<WeightableLayerTransformation> WeightableLayerTransformationPtr;

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
