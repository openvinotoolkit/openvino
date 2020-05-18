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

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(ConcatTransformation) : public LayerTransformation {
public:
    ConcatTransformation(const Params& params) : LayerTransformation(params) {}
    ~ConcatTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    static bool getQuantizeLayers(
        CNNLayerPtr layer,
        std::vector<std::string>& childNameOurAfterQuantizeLayers,
        std::vector<CNNLayerPtr>& quantizeLayers,
        std::vector<std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>>>& intermediateLayers,
        std::vector<CNNLayerPtr>& concatLayers,
        CNNLayerPtr child,
        std::vector<CNNLayerPtr>& sideOutputLayers,
        std::vector<std::string>& childrenNameSideOutputLayers);

protected:
    void addDequantizationForQuantize(
        TransformationContext& context,
        const CNNLayer& concat,
        const std::vector<CNNLayerPtr>& quantizeLayers,
        const std::vector<std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>>>& intermediateLayers,
        const std::vector<std::string>& childNameOurAfterQuantizeLayers,
        const std::unordered_map<std::string, std::vector<float>>& dequantizationScalesLayers,
        const std::unordered_map<std::string, std::vector<float>>& dequantizationShiftsLayers) const;

private:
    size_t getMinQuantizationLevels(
        const DataPrecision& dataPrecision,
        const float maxOutputInterval,
        const std::vector<QuantizationDetails>& quantizationLayersDetails,
        const float outputLowValue,
        const float outputHighValue) const;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
