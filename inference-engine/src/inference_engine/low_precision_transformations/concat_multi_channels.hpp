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

class INFERENCE_ENGINE_API_CLASS(ConcatMultiChannelsTransformation) : public ConcatTransformation {
private:
public:
    ConcatMultiChannelsTransformation(const Params& params) : ConcatTransformation(params) {}
    ~ConcatMultiChannelsTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    static bool getQuantizeLayers(
        CNNLayerPtr layer,
        std::vector<std::string>& childNameOurAfterQuantizeLayers,
        std::vector<CNNLayerPtr>& quantizeLayers,
        std::vector<std::vector<CNNLayerPtr>>& intermediateLayers,
        std::vector<CNNLayerPtr>& concatLayers,
        std::string childName,
        std::vector<CNNLayerPtr>& sideOutputLayers,
        std::vector<std::string>& childrenNameSideOutputLayers);
};

}  // namespace details
}  // namespace InferenceEngine
