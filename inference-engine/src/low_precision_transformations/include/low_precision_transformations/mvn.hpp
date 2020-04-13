// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include "layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(MvnTransformation) : public LayerTransformation {
public:
    MvnTransformation(const Params& params) : LayerTransformation(params) {}
    ~MvnTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
