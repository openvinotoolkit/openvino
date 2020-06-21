// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "ie_blob.h"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"

#include "cldnn_custom_layer.h"

#include <api/network.hpp>

namespace CLDNNPlugin {

struct Config {
    enum LptVersion {
        cnnNetwork,
        nGraph
    };

    Config() : throughput_streams(1),
               useProfiling(false),
               dumpCustomKernels(false),
               exclusiveAsyncRequests(false),
               memory_pool_on(true),
               enableDynamicBatch(false),
               enableInt8(true),
               nv12_two_inputs(false),
               queuePriority(cldnn::priority_mode_types::disabled),
               queueThrottle(cldnn::throttle_mode_types::disabled),
               max_dynamic_batch(1),
               customLayers({}),
               tuningConfig(),
               graph_dumps_dir(""),
               sources_dumps_dir(""),
               device_id("") {
        adjustKeyMapValues();
    }

    void UpdateFromMap(const std::map<std::string, std::string>& configMap);
    void adjustKeyMapValues();

    uint16_t throughput_streams;
    bool useProfiling;
    bool dumpCustomKernels;
    bool exclusiveAsyncRequests;
    bool memory_pool_on;
    bool enableDynamicBatch;
    bool enableInt8;
    LptVersion lptVersion = LptVersion::cnnNetwork;
    bool nv12_two_inputs;
    cldnn::priority_mode_types queuePriority;
    cldnn::throttle_mode_types queueThrottle;
    int max_dynamic_batch;
    CLDNNCustomLayerMap customLayers;
    cldnn::tuning_config_options tuningConfig;
    std::string graph_dumps_dir;
    std::string sources_dumps_dir;
    std::string device_id;

    std::map<std::string, std::string> key_config_map;
};

}  // namespace CLDNNPlugin
