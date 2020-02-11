// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include "ie_layers.h"
#include "low_precision_transformations/transformation_context.hpp"
#include "low_precision_transformations/layer_transformation.hpp"
#include "low_precision_transformations/transparent_base_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(PoolingTransformation) : public TransparentBaseTransformation {
public:
    PoolingTransformation(const Params& params) : TransparentBaseTransformation(params) {}
    ~PoolingTransformation() override {}
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
};

}  // namespace details
}  // namespace InferenceEngine
