// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <ie_common.h>
#include <algorithm>
#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(FakeQuantizeTransformation) : public LayerTransformation {
public:
    FakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {}
    ~FakeQuantizeTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    void setWeightsToConst(const bool weightsToConst);
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;

protected:
    void fuseScaleShift(TransformationContext& context, CNNLayerPtr fakeQuantizeLayer, CNNLayerPtr scaleShift) const;

    static Blob::Ptr reshapeWeightsIntervalConst(
        CNNLayer& constLayer,
        const std::vector<size_t>& dims,
        const Layout layout);

    static void reshapeFakeQuantize(
        CNNLayer& fakeQuantizeLayer,
        const std::vector<size_t>& dims,
        const Layout layout);
};

}  // namespace details
}  // namespace InferenceEngine
