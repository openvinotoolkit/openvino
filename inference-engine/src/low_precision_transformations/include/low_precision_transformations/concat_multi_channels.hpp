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
private:
public:
    ConcatMultiChannelsTransformation(const Params& params) : ConcatTransformation(params) {}
    ~ConcatMultiChannelsTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    static bool getQuantizeLayers(
        CNNLayerPtr layer,
        std::vector<std::string>& childNameOurAfterQuantizeLayers,
        std::vector<CNNLayerPtr>& quantizeLayers,
        std::vector<std::vector<std::pair<CNNLayerPtr, CNNLayerPtr> > >& intermediateLayers,
        std::vector<CNNLayerPtr>& concatLayers,
        CNNLayerPtr child,
        std::vector<CNNLayerPtr>& sideOutputLayers,
        std::vector<std::string>& childrenNameSideOutputLayers);
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
