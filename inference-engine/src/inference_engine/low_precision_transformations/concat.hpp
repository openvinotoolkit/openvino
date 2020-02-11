// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <ie_common.h>
#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(ConcatTransformation) : public LayerTransformation {
private:
public:
    ConcatTransformation(const Params& params) : LayerTransformation(params) {}
    ~ConcatTransformation() override {};
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

private:
    size_t getMinQuantizationLevels(
        const DataPrecision& dataPrecision,
        const float maxOutputInterval,
        const std::vector<QuantizationDetails>& quantizationLayersDetails,
        const float outputLowValue) const;
};

}  // namespace details
}  // namespace InferenceEngine
