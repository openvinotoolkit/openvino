// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "low_precision_transformations/eltwise.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(EltwiseCpuTransformation) : public EltwiseTransformation {
public:
    EltwiseCpuTransformation(const Params& params) : EltwiseTransformation(params) {}
    ~EltwiseCpuTransformation() override {}
    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
    void transform(TransformationContext& context, CNNLayer& layer) const override;

    bool isIncreasingTensor(const CNNLayer& layer) const;
private:
    static int getNotEmpty(const CNNLayer& eltwise);
};

}  // namespace details
}  // namespace InferenceEngine
