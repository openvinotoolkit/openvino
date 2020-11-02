// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(ResampleTransformation) : public LayerTransformation {
public:
    ResampleTransformation(const Params& params) : LayerTransformation(params) {}
    ~ResampleTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;
};

}  // namespace details
}  // namespace InferenceEngine
