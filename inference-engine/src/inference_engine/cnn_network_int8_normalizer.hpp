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
private:
    static void fillInScaleShift(ScaleShiftLayer* scshLayer, size_t c, float* weightsN, float* weightsD);

public:
    void NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats);

protected:
    void AddLayerToCNNNetworkBeforeLayer(CNNLayer::Ptr newLayer, CNNLayer::Ptr successor);
    void AddLayerToCNNNetworkAfterData(DataPtr pData, CNNLayer::Ptr layer, const std::string& nextLayerName);


    /**
     * Adds ScaleShift between two specified layers
     */
    void AddScaleShiftBetween(CNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2);
    /**
     * Adds ScaleShifts everywhere
     */
    void AddScaleShifts(CNNNetwork& net);

    /**
     * Converts the CNNNetwork from FP32 to Int8
     */
    void ConvertToInt8(int maxSign, int maxUnsign, CNNNetwork& net, const std::map<std::string, NetworkNodeStatsPtr>& netNodesStats);

    /**
     * Merging statistics from multiple sources.
     * The target statistics has max[i] = max(max1[i], max2[i]) and min[i] = min(min1[i], min2[i])
     */
    NetworkNodeStatsPtr mergeNetworkNodesStats(std::vector<NetworkNodeStatsPtr> stats);

    /**
     * Calculates a scale factor from statistics
     */
    InferenceEngine::Blob::Ptr calculateScaleFactor(const std::string& name, size_t channels, std::vector<NetworkNodeStatsPtr> stats, int maxInt);


    void PropagateScaleFactors(CNNNetwork& net);
    void ScaleDataToInt(const float* srcData, size_t srcSize, Blob::Ptr int8blob, const std::vector<float>& scales);
};

typedef std::shared_ptr<CNNNetworkInt8Normalizer> CNNNetworkNormalizerPtr;

}  // namespace details
}  // namespace InferenceEngine
