// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>

#include <ie_core.hpp>
#include <ie_icnn_network_stats.hpp>

class NetworkStatsCollector {
public:
    NetworkStatsCollector(const InferenceEngine::Core & ie, const std::string & deviceName);
    ~NetworkStatsCollector();

public:
    void ReadNetworkAndSetWeights(const void *model, size_t size, const InferenceEngine::TBlob<uint8_t>::Ptr &weights, size_t batch);
    void LoadNetwork(const std::string& modelPath, size_t batch);

    void InferAndCollectStats(const std::vector<std::string>& images,
                              std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);

/*    void InferAndCollectHistogram(const std::vector<std::string>& images,
                              const std::vector<std::string>& layerNames,
                              std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);

    void InferAndFindOptimalThreshold(const std::vector<std::string>& images,
                                  const std::vector<std::string>& layerNames,
                                  std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);

    void CalculateThreshold(std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);*/

    void CalculatePotentialMax(const float* weights, const InferenceEngine::SizeVector& weightDism, float& max);
    static InferenceEngine::CNNLayerPtr addScaleShiftBeforeLayer(std::string name, InferenceEngine::CNNLayer::Ptr beforeLayer,
            size_t port, std::vector<float> scale);

private:
    InferenceEngine::Core _ie;
    InferenceEngine::CNNNetwork _network;
    std::string _deviceName;
};
