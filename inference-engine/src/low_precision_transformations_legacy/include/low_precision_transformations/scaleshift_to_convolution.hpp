// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>
#include "low_precision_transformations/weightable_layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(ScaleShiftToConvolutionTransformation) : public WeightableLayerTransformation {
public:
    ScaleShiftToConvolutionTransformation(const Params& params);
    ~ScaleShiftToConvolutionTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;

    void setGroupSize(const size_t groupSize);
    size_t getGroupSize() const;

    void setIgnoreWithParents(const std::unordered_set<std::string>& ignoreWithParents);
    std::unordered_set<std::string> getIgnoreWithParents() const;

    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
    bool isQuantized(const CNNLayer& layer) const noexcept override;

private:
    CNNLayerPtr transformToConvolution(TransformationContext& context, const CNNLayer& layer, const size_t group) const;

    size_t groupSize;
    std::unordered_set<std::string> ignoreWithParents;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
