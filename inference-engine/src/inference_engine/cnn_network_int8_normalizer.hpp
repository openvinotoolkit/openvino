// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <float.h>

#include <string>
#include <vector>

#include <ie_icnn_network.hpp>
#include <ie_icnn_network_stats.hpp>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(CNNNetworkInt8Normalizer) {
public:
    CNNNetworkInt8Normalizer() {
    }

public:
    void NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats);

protected:
    void AddLayerToCNNNetwork(CNNNetwork& net, CNNLayerPtr firstNode, CNNLayerPtr secondNode, CNNLayerPtr nodeToInsert);
    void AddScaleShiftBeforeAndAfterInt8(CNNNetwork& net);
    void ConvertToInt8(int maxSign, int maxUnsign, CNNNetwork& net, const std::map<std::string, NetworkNodeStatsPtr>& netNodesStats);
    void ScaleDataToInt8(const float* srcData, size_t srcSize, Blob::Ptr int8blob, float maxValue, const std::vector<float>& scales);
};

typedef std::shared_ptr<CNNNetworkInt8Normalizer> CNNNetworkNormalizerPtr;

}  // namespace details
}  // namespace InferenceEngine
