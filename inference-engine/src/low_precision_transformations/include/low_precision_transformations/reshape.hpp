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

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(ReshapeTransformation) : public TransparentBaseTransformation {
public:
    ReshapeTransformation(const Params& params) : TransparentBaseTransformation(params) {}
    ~ReshapeTransformation() override {}
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;

private:
    bool canTransformOriginal(const CNNLayer& layer) const;
    void transformOriginal(TransformationContext& context, CNNLayer& layer) const;
    bool canTransformConstPropagated(const CNNLayer& layer) const;
    void transformConstPropagated(TransformationContext& context, CNNLayer& layer) const;
    void quantize(TransformationContext& context, CNNLayer& layer) const;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
