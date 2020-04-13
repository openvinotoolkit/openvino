// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_stats_impl.hpp"

#include <ie_common.h>

#include <cassert>
#include <cfloat>
#include <fstream>
#include <map>
#include <memory>
#include <pugixml.hpp>
#include <string>
#include <vector>

using namespace std;
namespace InferenceEngine {
namespace details {

CNNNetworkStatsImpl::~CNNNetworkStatsImpl() {}

void CNNNetworkStatsImpl::setNodesStats(const NetworkStatsMap& stats) {
    netNodesStats = stats;
}

const NetworkStatsMap& CNNNetworkStatsImpl::getNodesStats() const {
    return netNodesStats;
}

}  // namespace details
}  // namespace InferenceEngine
