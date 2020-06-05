// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <ie_common.h>

#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/layer_transformation.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(ConcatTransformation) : public LayerTransformation {
public:
    ConcatTransformation(const Params& params) : LayerTransformation(params) {}
    ~ConcatTransformation() override {};
    void transform(TransformationContext& context, CNNLayer& layer) const override;

protected:
    void addDequantizationLayers(
        TransformationContext& context,
        Subgraph& subgraph,
        std::function<void(
            const CNNLayer& layer,
            const std::string& originalLayerName,
            std::vector<float>& dequantizationScales,
            std::vector<float>& dequantizationShifts)> getLayerDequantizationCallback) const;

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
