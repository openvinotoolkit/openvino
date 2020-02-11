// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>

#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(EltwiseTransformation) : public LayerTransformation {
public:
    EltwiseTransformation(const Params& params) : LayerTransformation(params) {}
    ~EltwiseTransformation() override {};

    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
    void transform(TransformationContext& context, CNNLayer& layer) const override;

    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;

    static const int INTERVALS_THRESHOLD = 0;

    static bool isSupported(const TensorDesc& tensorDesc1, const TensorDesc& tensorDesc2);
    static bool isBroadcasted(const TensorDesc& tensorDesc);

private:
    static size_t getMinQuantizationLevels(const DataPrecision& dataPrecision, const std::vector<float>& outputIntervals);
};

}  // namespace details
}  // namespace InferenceEngine
