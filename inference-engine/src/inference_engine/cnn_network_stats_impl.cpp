// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>
#include "cnn_network_stats_impl.hpp"
#include <memory>
#include <map>
#include <string>
#include <fstream>
#include <cassert>
#include <cfloat>
#include "debug.h"
#include <vector>

#include <pugixml.hpp>

using namespace std;
namespace InferenceEngine {
namespace details {

CNNNetworkStatsImpl::~CNNNetworkStatsImpl() {
}

string joinCommas(vector<float>& v) {
    string res;

    for (size_t i = 0; i < v.size(); ++i) {
        res += to_string(v[i]);
        if (i < v.size() - 1) {
            res += ", ";
        }
    }

    return res;
}

void CNNNetworkStatsImpl::setNodesStats(const NetworkStatsMap &stats) {
    netNodesStats = stats;
}

const NetworkStatsMap& CNNNetworkStatsImpl::getNodesStats() const {
    return netNodesStats;
}

}  // namespace details
}  // namespace InferenceEngine
