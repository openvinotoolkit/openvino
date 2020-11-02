// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>
#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(NormalizeTransformation) : public LayerTransformation {
public:
    NormalizeTransformation(const Params& params) : LayerTransformation(params) {}
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
};

}  // namespace details
}  // namespace InferenceEngine
