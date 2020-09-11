// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_weights_cache.hpp"

#include <ie_system_conf.h>
#include <memory>

namespace MKLDNNPlugin {

const SimpleDataHash MKLDNNWeightsSharing::simpleCRC;

NumaNodesWeights::NumaNodesWeights() {
    for (auto numa_id : InferenceEngine::getAvailableNUMANodes())
        _cache_map[numa_id] = std::make_shared<MKLDNNWeightsSharing>();
}

MKLDNNWeightsSharing::Ptr& NumaNodesWeights::operator[](int numa_id) {
    auto found = _cache_map.find(numa_id);
    if (found == _cache_map.end())
        THROW_IE_EXCEPTION << "Unknown numa node id " << numa_id;
    return found->second;
}

const MKLDNNWeightsSharing::Ptr& NumaNodesWeights::operator[](int numa_id) const {
    auto found = _cache_map.find(numa_id);
    if (found == _cache_map.end())
        THROW_IE_EXCEPTION << "Unknown numa node id " << numa_id;
    return found->second;
}

}  // namespace MKLDNNPlugin
