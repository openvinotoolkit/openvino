// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ICNNNetworkStats class
 *
 * @file ie_icnn_network_stats.hpp
 */
#pragma once

#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <map>
#include "details/ie_irelease.hpp"

namespace InferenceEngine {

class NetworkNodeStats;
/**
 * @brief A shared pointer to the NetworkNodeStats object
 */
using NetworkNodeStatsPtr = std::shared_ptr<NetworkNodeStats>;
/**
 * @brief A smart pointer to the NetworkNodeStats object
 */
using NetworkNodeStatsWeakPtr = std::weak_ptr<NetworkNodeStats>;
/**
 * @brief A map of pairs: name of a layer and related statistics
 */
using NetworkStatsMap = std::map<std::string, NetworkNodeStatsPtr>;
/**
 * @class ICNNNetworkStats
 * @brief This is the interface to describe the NN topology scoring statistics
 */
class ICNNNetworkStats : public details::IRelease {
public:
    /**
     * @brief Sets a map which contains layers with statistics
     *
     * @param stats A map which is set
     * Abstract method
     */
    virtual void setNodesStats(const NetworkStatsMap& stats) = 0;
    /**
     * @brief Gets a map which contains layers with statistics
     *
     * Abstract method
     * @return A NetworkStatsMap object
     */
    virtual const NetworkStatsMap& getNodesStats() const = 0;
    /**
     * @brief Checks if a container is empty
     *
     * Abstract method
     * @return A bool value which shows whether a container is empty
     */
    virtual bool isEmpty() const = 0;
};

/**
 * @class NetworkNodeStats
 * @brief This class implements a container which stores statistics for a layer
 */
class NetworkNodeStats {
public:
    /**
     * @brief The constructor which creates NetworkNodeStats object
     */
    NetworkNodeStats() { }
    /**
     * @brief The constructor which creates NetworkNodeStats object with filled statistics
     *
     * @param statCount The number of minimum/maximum values in statistics
     */
    explicit NetworkNodeStats(int statCount) {
        float mn = (std::numeric_limits<float>::max)();
        float mx = (std::numeric_limits<float>::min)();

        for (int i = 0; i < statCount; i++) {
            _minOutputs.push_back(mn);
            _maxOutputs.push_back(mx);
        }
    }

public:
    /**
     * @brief Vector of floats which contains minimum values of layers activations
     */
    std::vector<float> _minOutputs;
    /**
     * @brief Vector of floats which contains maximum values of layers activations
     */
    std::vector<float> _maxOutputs;
};


}  // namespace InferenceEngine
