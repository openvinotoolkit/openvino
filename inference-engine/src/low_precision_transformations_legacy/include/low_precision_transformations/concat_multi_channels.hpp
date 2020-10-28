// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <ie_common.h>
#include "low_precision_transformations/concat.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(ConcatMultiChannelsTransformation) : public ConcatTransformation {
public:
    ConcatMultiChannelsTransformation(const Params& params) : ConcatTransformation(params) {}
    ~ConcatMultiChannelsTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;

private:
    static void fillDequantization(
        const CNNLayer& layer,
        const std::unordered_map<std::string, std::vector<float>>& dequantizationScalesLayers,
        const std::unordered_map<std::string, std::vector<float>>& dequantizationShiftsLayers,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts);

    static void fillQuantization(const CNNLayer& layer, std::vector<CNNLayerPtr>& fakeQuantizes);
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
