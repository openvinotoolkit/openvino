// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <ie_icnn_network.hpp>
#include <cpp/ie_cnn_network.h>
#include "low_precision_transformations/quantization_details.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class TransformationContext {
public:
    explicit TransformationContext(ICNNNetwork& network);

    void removeLayer(const CNNLayer& layer);
    ICNNNetwork& network;
    std::unordered_set<std::string> quantizedFakeQuantizeNames;
    std::unordered_set<std::string> dequantizationLayersNames;

    const std::vector<CNNLayerPtr>& getLayers() {
        return layers;
    }

    inline Precision getOriginalLayerPrecision(const std::string& layer_name, const std::string& data_name = "") {
        const auto& data_map = _original_precisions_map.find(layer_name);
        if (data_map == _original_precisions_map.end())
            return Precision::UNSPECIFIED;
        if (data_name.empty() && data_map->second.size() > 0)
            return data_map->second.begin()->second;
        if (data_map->second.find(data_name) == data_map->second.end())
            return Precision::UNSPECIFIED;
        return data_map->second[data_name];
    }

private:
    std::vector<CNNLayerPtr> layers;
    std::unordered_map<std::string, std::unordered_map<std::string, Precision>> _original_precisions_map;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
