// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(EltwiseTransformation) : public LayerTransformation {
public:
    EltwiseTransformation(const Params& params) : LayerTransformation(params) {}
    ~EltwiseTransformation() override {}
    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
    void transform(TransformationContext& context, CNNLayer& layer) const override;

    bool isBroadcastByChannels(const CNNLayer& layer) const;

    static bool isSupported(const TensorDesc& tensorDesc1, const TensorDesc& tensorDesc2) noexcept;
    static bool isBroadcasted(const TensorDesc& tensorDesc) noexcept;

private:
    static int getNotEmpty(const CNNLayer& eltwise);
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
