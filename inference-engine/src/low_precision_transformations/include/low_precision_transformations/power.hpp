// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <legacy/ie_layers.h>
#include "low_precision_transformations/transformation_context.hpp"
#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(PowerTransformation) : public LayerTransformation {
public:
    PowerTransformation(const Params& params) : LayerTransformation(params) {}
    ~PowerTransformation() override {}
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
};

}  // namespace details
}  // namespace InferenceEngine
