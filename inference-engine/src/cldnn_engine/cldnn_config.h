// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "ie_blob.h"
#include "ie_plugin.hpp"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"
#include "inference_engine.hpp"

#include "cldnn_custom_layer.h"

#include <CPP/network.hpp>

namespace CLDNNPlugin {

struct Config {
    Config() : throughput_streams(1),
               useProfiling(false),
               dumpCustomKernels(false),
               exclusiveAsyncRequests(false),
               memory_pool_on(true),
               enableDynamicBatch(false),
               enableInt8(false),
               queuePriority(cldnn::priority_mode_types::disabled),
               queueThrottle(cldnn::throttle_mode_types::disabled),
               max_dynamic_batch(1),
               customLayers({}),
               tuningConfig(),
               graph_dumps_dir(""),
               sources_dumps_dir("") {
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
    cldnn::priority_mode_types queuePriority;
    cldnn::throttle_mode_types queueThrottle;
    int max_dynamic_batch;
    CLDNNCustomLayerMap customLayers;
    cldnn::tuning_config_options tuningConfig;
    std::string graph_dumps_dir;
    std::string sources_dumps_dir;

    std::map<std::string, std::string> key_config_map;
};

}  // namespace CLDNNPlugin
