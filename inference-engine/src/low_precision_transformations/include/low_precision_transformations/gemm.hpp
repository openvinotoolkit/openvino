// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <ie_common.h>
#include <algorithm>
#include "low_precision_transformations/layer_transformation.hpp"
#include "low_precision_transformations/fully_connected.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(GemmTransformation) : public FullyConnectedTransformation {
public:
    GemmTransformation(const LayerTransformation::Params& params) : FullyConnectedTransformation(params) {}
    ~GemmTransformation() override {};
    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
    void transform(TransformationContext& context, CNNLayer& layer) const override;

    bool isQuantized(const CNNLayer& layer) const noexcept override;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
